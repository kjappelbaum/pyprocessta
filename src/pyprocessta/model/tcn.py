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