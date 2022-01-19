# -*- coding: utf-8 -*-
from functools import partial
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.logging import raise_if_not
from darts.models import TCNModel
from darts.utils import _build_tqdm_iterator
from darts.utils.data.inference_dataset import InferenceDataset
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from .utils import enable_dropout

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
TARGETS = ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"]


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
        past_covariates=x_test,
        start=start,
        forecast_horizon=horizon,
        stride=stride,
        retrain=False,
        verbose=False,
    )
    return backtest


def parallelized_inference(model, x, y, repeats=100, start=0.3, stride=1, horizon=6):
    results = []
    model.model.eval()
    enable_dropout(model.model)

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


def get_data(
    df: pd.DataFrame, targets: List[str] = TARGETS, features: List[str] = MEAS_COLUMNS
) -> tuple:
    """Build x (covariates) and y (targets) time series from dataframe

    Args:
        df (pd.Dataframe): Must be time-indexed
        targets (List[str], optional): names of target columns.
            Defaults to TARGETS.
        features (List[str], optional): names of covariate columns.
            Defaults to MEAS_COLUMNS.

    Returns:
        tuple: [description]
    """
    y = TimeSeries.from_dataframe(df, value_cols=targets)
    x = TimeSeries.from_dataframe(df, value_cols=features)
    return x, y


def transform_data(train_tuple: tuple, test_tuples: List[tuple]):
    """Scale data using minmax scaling

    Args:
        train_tuple (tuple): tuple of darts time series for training
        test_tuples (List[tuple]): tuples (x,y) of darts time series for testing

    Returns:
        tuple: tuple of time series for training, test tuples and transformers
    """
    x_train, y_train = train_tuple

    transformer = Scaler()

    x_train = transformer.fit_transform(x_train)

    y_transformer = Scaler(name="YScaler")
    y_train = y_transformer.fit_transform(y_train)

    transformed_test_tuples = []
    for x_test, y_test in test_tuples:
        print(x_test.pd_dataframe().shape, y_test.pd_dataframe().shape)
        x_test = transformer.transform(x_test)
        y_test = y_transformer.transform(y_test)
        transformed_test_tuples.append((x_test, y_test))

    return (x_train, y_train), transformed_test_tuples, (transformer, y_transformer)


def get_train_test_data(
    x: TimeSeries, y: TimeSeries, split_date="2010-01-18 12:59:15"
) -> tuple:
    """Perform a train/test split at given data

    Args:
        x (TimeSeries): darts TimeSeries object
        y (TimeSeries): darts Timeseries object
        split_date (str, optional): Date at which split is performed.
            Data before this date is used for training tuple.
            Data after this date for the testing tuple.
            Defaults to "2010-01-18 12:59:15".

    Returns:
        tuple: tuples of x,y for train and test set
    """
    y_train, y_test = y.split_before(pd.Timestamp(split_date))
    x_train, x_test = x.split_before(pd.Timestamp(split_date))

    return (x_train, y_train), (x_test, y_test)


class TCNModelDropout(TCNModel):
    def predict_from_dataset(
        self,
        n: int,
        input_series_dataset: InferenceDataset,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
    ):
        return self.predict_from_dataset(
            n,
            input_series_dataset,
            batch_size,
            verbose,
            n_jobs,
            roll_size,
            num_samples,
            num_loader_workers,
        )


def run_model(
    train_tuple: tuple,
    input_chunk_length: int = 60,
    output_chunk_length: int = 30,
    num_layers: int = 16,
    num_filters: int = 8,
    kernel_size: int = 4,
    dropout: float = 0.5627,
    weight_norm: bool = True,
    batch_size: int = 128,
    n_epochs: int = 200,
    lr: float = 0.02382,
) -> TCNModelDropout:

    x_train, y_train = train_tuple
    model_cov = TCNModelDropout(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_layers=num_layers,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout=dropout,
        weight_norm=weight_norm,
        batch_size=batch_size,
        n_epochs=n_epochs,
        log_tensorboard=True,
        optimizer_kwargs={"lr": lr},
    )

    model_cov.fit(series=y_train, past_covariates=x_train, verbose=False)

    return model_cov


class TCNModelDropout(TCNModel):
    def predict_with_dropout(
        self,
        n: int,
        input_series_dataset: InferenceDataset,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
    ):
        self.model.eval()
        enable_dropout(self)
        return self.predict_from_dataset(
            n,
            input_series_dataset,
            batch_size,
            verbose,
            n_jobs,
            roll_size,
            num_samples,
            num_loader_workers,
        )

    def predict_from_dataset(
        self,
        n: int,
        input_series_dataset: InferenceDataset,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
    ) -> Sequence[TimeSeries]:

        """
        This method allows for predicting with a specific :class:`darts.utils.data.InferenceDataset` instance.
        These datasets implement a PyTorch `Dataset`, and specify how the target and covariates are sliced
        for inference. In most cases, you'll rather want to call :func:`predict()` instead, which will create an
        appropriate :class:`InferenceDataset` for you.
        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        input_series_dataset
            Optionally, a series or sequence of series, representing the history of the target series' whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        batch_size
            Size of batches during prediction. Defaults to the models `batch_size` value.
        verbose
            Shows the progress bar for batch predicition. Off by default.
        n_jobs
            The number of jobs to run in parallel. `-1` means using all processors. Defaults to `1`.
        roll_size
            For self-consuming predictions, i.e. `n > output_chunk_length`, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set `output_chunk_length` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        num_loader_workers
            Optionally, an integer specifying the `num_workers` to use in PyTorch ``DataLoader`` instances,
            for the inference/prediction dataset loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.
        Returns
        -------
        Sequence[TimeSeries]
            Returns one or more forecasts for time series.
        """
        self._verify_inference_dataset_type(input_series_dataset)

        # check that covariates and dimensions are matching what we had during training
        self._verify_predict_sample(input_series_dataset[0])

        if roll_size is None:
            roll_size = self.output_chunk_length
        else:
            raise_if_not(
                0 < roll_size <= self.output_chunk_length,
                "`roll_size` must be an integer between 1 and `self.output_chunk_length`.",
            )

        # check that `num_samples` is a positive integer
        raise_if_not(num_samples > 0, "`num_samples` must be a positive integer.")

        # iterate through batches to produce predictions
        batch_size = batch_size or self.batch_size

        pred_loader = DataLoader(
            input_series_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_fn,
        )
        predictions = []
        iterator = _build_tqdm_iterator(pred_loader, verbose=verbose)

        #self.model.eval()
        with torch.no_grad():
            for batch_tuple in iterator:
                batch_tuple = self._batch_to_device(batch_tuple)
                input_data_tuple, batch_input_series = batch_tuple[:-1], batch_tuple[-1]

                # number of individual series to be predicted in current batch
                num_series = input_data_tuple[0].shape[0]

                # number of of times the input tensor should be tiled to produce predictions for multiple samples
                # this variable is larger than 1 only if the batch_size is at least twice as large as the number
                # of individual time series being predicted in current batch (`num_series`)
                batch_sample_size = min(max(batch_size // num_series, 1), num_samples)

                # counts number of produced prediction samples for every series to be predicted in current batch
                sample_count = 0

                # repeat prediction procedure for every needed sample
                batch_predictions = []
                while sample_count < num_samples:

                    # make sure we don't produce too many samples
                    if sample_count + batch_sample_size > num_samples:
                        batch_sample_size = num_samples - sample_count

                    # stack multiple copies of the tensors to produce probabilistic forecasts
                    input_data_tuple_samples = self._sample_tiling(
                        input_data_tuple, batch_sample_size
                    )

                    # get predictions for 1 whole batch (can include predictions of multiple series
                    # and for multiple samples if a probabilistic forecast is produced)
                    batch_prediction = self._get_batch_prediction(
                        n, input_data_tuple_samples, roll_size
                    )

                    # reshape from 3d tensor (num_series x batch_sample_size, ...)
                    # into 4d tensor (batch_sample_size, num_series, ...), where dim 0 represents the samples
                    out_shape = batch_prediction.shape
                    batch_prediction = batch_prediction.reshape(
                        (
                            batch_sample_size,
                            num_series,
                        )
                        + out_shape[1:]
                    )

                    # save all predictions and update the `sample_count` variable
                    batch_predictions.append(batch_prediction)
                    sample_count += batch_sample_size

                # concatenate the batch of samples, to form num_samples samples
                batch_predictions = torch.cat(batch_predictions, dim=0)
                batch_predictions = batch_predictions.cpu().detach().numpy()

                # create `TimeSeries` objects from prediction tensors
                ts_forecasts = Parallel(n_jobs=n_jobs)(
                    delayed(self._build_forecast_series)(
                        [
                            batch_prediction[batch_idx]
                            for batch_prediction in batch_predictions
                        ],
                        input_series,
                    )
                    for batch_idx, input_series in enumerate(batch_input_series)
                )

                predictions.extend(ts_forecasts)

        return predictions