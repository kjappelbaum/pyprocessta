from itertools import product
import concurrent.futures
import pickle
import pickle
import os

from numpy.lib.function_base import meshgrid
from sklearn import preprocessing
from definitions import MEASUREMENTS, TARGETS
from functools import partial


def load_pickle(filename):
    with open(filename, "rb") as handle:
        res = pickle.load(handle)
    return res


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from copy import deepcopy
from functools import partial
import concurrent.futures
import numpy as np
from collections import defaultdict
import click


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


DF = pd.read_pickle("df_dropped.pkl")

MEAS_COLUMNS = [
    "TI-19",
    "FI-16",
    "TI-33",
    "FI-2",
    "FI-151",
    "TI-8",
    "FI-241",
    "Level Desorber",
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
]

TARGETS.pop(0)

SCALER = load_pickle(os.path.join("x_scaler.pkl"))

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

    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as exec:
        for point, result in zip(meshgrid, map(run_target_wrapper, meshgrid)):
            print(point)
            point_a, point_b = point
            results_double[point_a][point_b] = result

    dump_pickle(results_double, f"{feature_1}_{feature_2}.pkl")


if __name__ == "__main__":
    main()
