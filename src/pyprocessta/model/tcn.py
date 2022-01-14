from darts import TimeSeries
import pandas as pd
from definitions import MEASUREMENTS, TARGETS
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
from pyprocessta.preprocess.resample import resample_regular
import wandb
from .utils import enable_dropout
import torch
from darts.logging import raise_if_not
from darts.utils.data.timeseries_dataset import TimeSeriesInferenceDataset
from typing import Sequence, Optional
import concurrent.futures
from functools import partial
from torch.optim.lr_scheduler import CyclicLR
import numpy as np
from torch.utils.data import DataLoader
from darts.utils import _build_tqdm_iterator
from joblib import Parallel, delayed
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
MEAS_COLUMNS = [
    "TI-19",
    "FI-16",
    "TI-33",
    "FI-2",
    "FI-151",
    "TI-8",
    "FI-241",
    "valve-position-12",  # dry-bed
    "FI-38",  # stripper
    "PI-28",  # stripper
    "TI-28",  # stripper
    "FI-20",
    "FI-30",
    "TI-3",
    "FI-19",
    "FI-211",
    "FI-11",
    "TI-30",
    "PI-30",
    "TI-1213",
    "TI-4",
    "FI-23",
    "delta_t",
]

df = pd.read_pickle("df_dropped.pkl")


def summarize_results(results):
    values = []

    for df in results:
        values.append(df.pd_dataframe().values)

    df = df.pd_dataframe()
    columns = df.columns

    return (
        pd.DataFrame(np.mean(values, axis=0), columns=columns, index=df.index),
        pd.DataFrame(np.std(values, axis=0), columns=columns, index=df.index),
    )


def _run_backtest(rep, model, x_test, y_test, start=0.3, stride=1, horizon=4):
    backtest = model.historical_forecasts(
        y_test,
        covariates=x_test,
        start=start,
        forecast_horizon=horizon,
        stride=stride,
        retrain=False,
        verbose=False,
    )
    return backtest


def parallelized_inference(model, x, y, repeats=100, start=0.3, stride=1, horizon=6):
    results = []

    backtest_partial = partial(
        _run_backtest,
        model=model,
        x_test=x,
        y_test=y,
        start=start,
        stride=stride,
        horizon=horizon,
    )

    for res in map(backtest_partial, range(repeats)):
        results.append(res)

    return results


def get_data():
    y = TimeSeries.from_dataframe(df, value_cols=TARGETS)
    x = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS)
    return x, y


def transform_data(train_tuple, test_tuples):
    x_train, y_train = train_tuple

    transformer = Scaler()

    x_train = transformer.fit_transform(x_train)

    # with open("x_scaler.pkl", "wb") as handle:
    #     pickle.dump(transformer, handle)

    y_transformer = Scaler(name="YScaler")
    y_train = y_transformer.fit_transform(y_train)

    transformed_test_tuples = []
    for x_test, y_test in test_tuples:
        print(x_test.pd_dataframe().shape, y_test.pd_dataframe().shape)
        x_test = transformer.transform(x_test)
        y_test = y_transformer.transform(y_test)
        transformed_test_tuples.append((x_test, y_test))

    # with open("y_scaler.pkl", "wb") as handle:
    #     pickle.dump(y_transformer, handle)

    return (x_train, y_train), transformed_test_tuples, (transformer, y_transformer)


def get_train_test_data(x, y, split_date="2010-01-18 12:59:15"):
    y_train, y_test = y.split_before(pd.Timestamp(split_date))
    x_train, x_test = x.split_before(pd.Timestamp(split_date))

    return (x_train, y_train), (x_test, y_test)


def run_model(train_tuple, input_chunk_length=40, output_chunk_length=10):

    # run = wandb.init(project='process_ml', reinit=True, sync_tensorboard=True)
    # with run:
    x_train, y_train = train_tuple
    model_cov = TCNModelDropout(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_layers=8,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
        weight_norm=True,
        batch_size=32,
        n_epochs=400,
        log_tensorboard=True,
        optimizer_kwargs={"lr": 5e-6},
        model_name=f"{timestr}-tcn"
    )

    model_cov.fit(series=y_train, covariates=x_train, verbose=False)

    return model_cov


class TCNModelDropout(TCNModel):
    # the batch arguments are currently not implemented and are just for compatibility of the API
    def predict_from_dataset(
        self,
        n: int,
        input_series_dataset: TimeSeriesInferenceDataset,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        n_jobs=1,
    ) -> Sequence[TimeSeries]:
        self.model.eval()
        enable_dropout(self)

        # preprocessing
        raise_if_not(
            isinstance(input_series_dataset, TimeSeriesInferenceDataset),
            "Only TimeSeriesInferenceDataset is accepted as input type",
        )

        # check that the input sizes match
        sample = input_series_dataset[0]

        in_dim = sum(map(lambda ts: (ts.width if ts is not None else 0), sample))
        raise_if_not(
            in_dim == self.input_dim,
            "The dimensionality of the series provided for prediction does not match the dimensionality "
            "of the series this model has been trained on. Provided input dim = {}, "
            "model input dim = {}".format(in_dim, self.input_dim),
        )

        # TODO currently we assume all forecasts fit in memory
        in_tsr_arr = []
        for target_series, covariate_series in input_series_dataset:
            raise_if_not(
                len(target_series) >= self.input_chunk_length,
                "All input series must have length >= `input_chunk_length` ({}).".format(
                    self.input_chunk_length
                ),
            )

            # TODO: here we could be smart and handle cases where target and covariates do not have same time axis.
            # TODO: e.g. by taking their latest common timestamp.

            in_tsr_sample = target_series.values(copy=False)[-self.input_chunk_length :]
            in_tsr_sample = torch.from_numpy(in_tsr_sample).float().to(self.device)
            if covariate_series is not None:
                in_cov_tsr = covariate_series.values(copy=False)[
                    -self.input_chunk_length :
                ]
                in_cov_tsr = torch.from_numpy(in_cov_tsr).float().to(self.device)
                in_tsr_sample = torch.cat([in_tsr_sample, in_cov_tsr], dim=1)
            in_tsr_sample = in_tsr_sample.view(1, self.input_chunk_length, -1)

            in_tsr_arr.append(in_tsr_sample)

        # concatenate to one tensor of size [len(input_series_dataset), input_chunk_length, 1 + # of covariates)]
        in_tsr = torch.cat(in_tsr_arr, dim=0)

        # prediction
        pred_loader = DataLoader(
            in_tsr,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        predictions = []

        iterator = _build_tqdm_iterator(pred_loader, verbose=verbose)

        with torch.no_grad():
            for batch in iterator:
                batch_prediction = []  # (num_batches, n % output_chunk_length)
                out = self.model(batch)[
                    :, self.first_prediction_index :, :
                ]  # (batch_size, output_chunk_length, width)
                batch_prediction.append(out)
                while sum(map(lambda t: t.shape[1], batch_prediction)) < n:
                    roll_size = min(self.output_chunk_length, self.input_chunk_length)
                    batch = torch.roll(batch, -roll_size, 1)
                    batch[:, -roll_size:, :] = out[:, :roll_size, :]
                    # take only last part of the output sequence where needed
                    out = self.model(batch)[:, self.first_prediction_index :, :]
                    batch_prediction.append(out)

                batch_prediction = torch.cat(batch_prediction, dim=1)
                batch_prediction = batch_prediction[:, :n, :]
                batch_prediction = batch_prediction.cpu().detach().numpy()

                ts_forecasts = Parallel(n_jobs=n_jobs)(
                    delayed(self._build_forecast_series)(prediction, input_series[0])
                    for prediction, input_series in zip(
                        batch_prediction, input_series_dataset
                    )
                )

                predictions.extend(ts_forecasts)

        return predictions
