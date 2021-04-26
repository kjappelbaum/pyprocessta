from darts import TimeSeries
import pandas as pd
from definitions import MEASUREMENTS, TARGETS
from .scaler import Scaler
from darts.models import TCNModel
from pyprocessta.preprocess.resample import resample_regular
import wandb
from .utils import enable_dropout
import torch
from darts.logging import raise_if_not
from darts.utils.data.timeseries_dataset import TimeSeriesInferenceDataset
from typing import Sequence
import concurrent.futures
from functools import partial
from torch.optim.lr_scheduler import CyclicLR

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


def _run_backtest(rep, model, x_test, y_test, start=0.3, stride=2):
    backtest = model.historical_forecasts(
        y_test,
        covariates=x_test,
        start=start,
        forecast_horizon=1,
        stride=stride,
        retrain=False,
        verbose=False,
    )
    return backtest


def parallelized_inference(model, x, y, repeats=100, start=0.3, stride=2):
    results = []

    backtest_partial = partial(
        _run_backtest, model=model, x_test=x, y_test=y, start=start, stride=stride
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

    y_transformer = Scaler()
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


def run_model(train_tuple):

    # run = wandb.init(project='process_ml', reinit=True, sync_tensorboard=True)
    # with run:
    x_train, y_train = train_tuple
    model_cov = TCNModelDropout(
        input_chunk_length=8,
        output_chunk_length=1,
        num_layers=5,
        num_filters=16,
        kernel_size=6,
        dropout=0.3,
        weight_norm=True,
        batch_size=32,
        n_epochs=100,
        log_tensorboard=True,
        optimizer_kwargs={"lr": 2e-4},
    )

    model_cov.fit(series=y_train, covariates=x_train, verbose=False)

    return model_cov


class TCNModelDropout(TCNModel):
    def predict_from_dataset(
        self, n: int, input_series_dataset: TimeSeriesInferenceDataset
    ) -> Sequence[TimeSeries]:
        self.model.eval()
        enable_dropout(self)
        # check that the input sizes match
        sample = input_series_dataset[0]
        in_dim = sum(map(lambda ts: (ts.width if ts is not None else 0), sample))
        raise_if_not(
            in_dim == self.input_dim,
            "The dimensionality of the series provided for prediction does not match the dimensionality"
            "of the series this model has been trained on. Provided input dim = {}, "
            "model input dim = {}".format(in_dim, self.input_dim),
        )

        # TODO use a torch Dataset and DataLoader for parallel loading and batching
        # TODO also currently we assume all forecasts fit in memory
        ts_forecasts = []
        for target_series, covariate_series in input_series_dataset:
            raise_if_not(
                len(target_series) >= self.input_chunk_length,
                "All input series must have length >= `input_chunk_length` ({}).".format(
                    self.input_chunk_length
                ),
            )

            # TODO: here we could be smart and handle cases where target and covariates do not have same time axis.
            # TODO: e.g. by taking their latest common timestamp.

            in_tsr = target_series.values(copy=False)[-self.input_chunk_length :]
            in_tsr = torch.from_numpy(in_tsr).float().to(self.device)
            if covariate_series is not None:
                in_cov_tsr = covariate_series.values(copy=False)[
                    -self.input_chunk_length :
                ]
                in_cov_tsr = torch.from_numpy(in_cov_tsr).float().to(self.device)
                in_tsr = torch.cat([in_tsr, in_cov_tsr], dim=1)
            in_tsr = in_tsr.view(1, self.input_chunk_length, -1)

            out_sequence = self._produce_prediction(in_tsr, n)

            # translate to numpy
            out_sequence = out_sequence.cpu().detach().numpy()
            ts_forecasts.append(
                self._build_forecast_series(
                    out_sequence.reshape(n, -1), input_series=target_series
                )
            )
        return ts_forecasts
