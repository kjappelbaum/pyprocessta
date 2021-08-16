# -*- coding: utf-8 -*-
import concurrent.futures
import os
import pickle
from functools import partial
from itertools import product

from definitions import MEASUREMENTS, TARGETS
from numpy.lib.function_base import meshgrid
from sklearn import preprocessing


def load_pickle(filename):
    with open(filename, "rb") as handle:
        res = pickle.load(handle)
    return res


import concurrent.futures
from collections import defaultdict
from copy import deepcopy
from functools import partial

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from darts import TimeSeries


def predict(x, y, model):
    backtest_cov = model.historical_forecasts(
        y,
        covariates=x,
        start=0.05,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True,
    )

    return backtest_cov.pd_dataframe()


DF = pd.read_pickle("20210508_df_cleaned.pkl")

MEAS_COLUMNS = [
    "TI-19",
    "valve-position-12",
    "TI-3",
    "PI-3",
    "FI-19",
    "FI-11",
    "TI-1213",
    "FI-23",
    "FI-20",
    "FI-20/FI-23",
    "delta_t",
]

TARGETS.pop(0)

SCALER = load_pickle(os.path.join("x_scaler_reduced_feature_set.pkl"))

X = TimeSeries.from_dataframe(DF, value_cols=MEAS_COLUMNS)

X = SCALER.transform(X)

FEAT_NUM_MAPPING = dict(zip(MEAS_COLUMNS, [str(i) for i in range(len(MEAS_COLUMNS))]))

UPDATE_MAPPING = {
    "target": {
        "scaler": load_pickle(os.path.join("y_scaler.pkl")),
        "model": load_pickle(os.path.join("multivariate_model.pkl")),
        "name": TARGETS,
    }
}


def run_update(x):

    model_dict = UPDATE_MAPPING["target"]
    y = model_dict["scaler"].transform(
        TimeSeries.from_dataframe(DF, value_cols=model_dict["name"])
    )
    df = predict(x, y, model_dict["model"])
    return df


def run_targets(feature_levels):

    df = deepcopy(DF)

    for k, v in feature_levels.items():
        df[k] = df[k] + v / 100 * np.abs(df[k])

    X_ = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS)
    X_ = SCALER.transform(X_)

    res = run_update(X_)
    return res


def dump_pickle(obj, filename):
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle)


@click.command("cli")
@click.argument("feature_1")
@click.argument("feature_2")
@click.argument("num_points", type=int, default=21)
def main(feature_1, feature_2, num_points):
    grid = np.linspace(-100, 100, num_points)
    results_double = defaultdict(dict)

    meshgrid = product(grid, grid)

    def run_target_wrapper(points):
        features = dict(zip([feature_1, feature_2], points))
        return run_targets(features)

    with concurrent.futures.ProcessPoolExecutor() as exec:
        for point, result in zip(meshgrid, exec.map(run_target_wrapper, meshgrid)):
            print(point_a, point_b)
            point_a, point_b = point
            results_double[point_a][point_b] = result

    dump_pickle(results_double, f"{feature_1}_{feature_2}.pkl")


if __name__ == "__main__":
    main()
