# -*- coding: utf-8 -*-
import os
import pickle
import time
from collections import defaultdict
from copy import deepcopy

import click
import joblib
import numpy as np
import pandas as pd
from darts import TimeSeries

from pyprocessta.model.tcn import (
    TCNModelDropout,
    parallelized_inference,
    summarize_results,
)

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def load_pickle(filename):
    with open(filename, "rb") as handle:
        res = pickle.load(handle)
    return res


def dump_pickle(object, filename):
    with open(filename, "wb") as handle:
        pickle.dump(object, handle)


MEAS_COLUMNS = [
    "TI-19",
    #      "FI-16",
    #     "TI-33",
    #     "FI-2",
    #     "FI-151",
    #     "TI-8",
    #     "FI-241",
    #  "valve-position-12",  # dry-bed
    #     "FI-38",  # strippera
    #     "PI-28",  # stripper
    #     "TI-28",  # stripper
    #      "FI-20",
    #     "FI-30",
    "TI-3",
    "FI-19",
    #     "FI-211",
    "FI-11",
    #     "TI-30",
    #     "PI-30",
    "TI-1213",
    #     "TI-4",
    #    "FI-23",
    #    "FI-20",
    #   "FI-20/FI-23",
    #    "TI-22",
    #    "delta_t",
    "TI-35",
    #     "delta_t_2"
]

# First, we train a model on *all* data
# Then, we do a partial-denpendency plot approach and change one variable and see how the model predictions change

# load the trained model

model_cov1 = TCNModelDropout(
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


model_cov2 = TCNModelDropout(
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


FEAT_NUM_MAPPING = dict(zip(MEAS_COLUMNS, [str(i) for i in range(len(MEAS_COLUMNS))]))
UPDATE_MAPPING = {
    "amine": {
        "scaler": joblib.load("20210814_2_y_transformer__reduced_feature_set"),
        "model": model_cov1.load_from_checkpoint(
            os.path.join(THIS_DIR, "20210814_2_amp_pip_model_reduced_feature_set_darts")
        ),
        "name": ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"],
    },
    "co2": {
        "scaler": joblib.load(
            "20210814_2_y_transformer_co2_ammonia_reduced_feature_set"
        ),
        "model": model_cov2.load_from_checkpoint(
            os.path.join(
                THIS_DIR, "20210814_2_co2_ammonia_model_reduced_feature_set_darts"
            )
        ),
        "name": ["Carbon dioxide CO2", "Ammonia NH3"],
    },
}
SCALER = joblib.load("20210814_2_x_scaler_reduced_feature_set")


# making the input one item longer as "safety margin"
def calculate_initialization_percentage(
    timeseries_length: int, input_sequence_length: int = 61
):
    fraction_of_input = input_sequence_length / timeseries_length
    res = max([fraction_of_input, 0.1])
    return res


def run_update(df, x, target="amine"):
    model_dict = UPDATE_MAPPING[target]
    y = model_dict["scaler"].transform(
        TimeSeries.from_dataframe(df, value_cols=model_dict["name"])
    )
    df = parallelized_inference(
        model_dict["model"],
        x,
        y,
        repeats=2,
        start=calculate_initialization_percentage(len(y)),
        horizon=2,
    )
    means, stds = summarize_results(df)
    return {"means": means, "stds": stds}


def run_targets(df, feature_levels, target: str = "amine"):

    df = deepcopy(df)

    for k, v in feature_levels.items():
        if k == "valve-position-12":
            df[k] = v
        else:
            df[k] = df[k] + v / 100 * np.abs(df[k])

    X_ = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS)
    X_ = SCALER.transform(X_)

    res = run_update(df, X_, target)
    return res


def run_grid(
    df,
    feature_a: str = "TI-19",
    feature_b: str = "FI-19",
    lower: float = -20,
    upper: float = 20,
    num_points: int = 4,
    objectives: str = "amine",
):
    grid = np.linspace(lower, upper, num_points)
    results_double_new = defaultdict(dict)

    if feature_a == "valve-position-12":
        grid_a = [0, 1]
    else:
        grid_a = grid
    if feature_b == "valve-position-12":
        grid_b = [0, 1]
    else:
        grid_b = grid

    for point_a in grid_a:
        for point_b in grid_b:
            print(f"Running point {feature_a}: {point_a} {feature_b}: {point_b}")
            results_double_new[point_a][point_b] = run_targets(
                df, {feature_a: point_a, feature_b: point_b}, objectives
            )
    return results_double_new


@click.command("cli")
@click.argument("feature_a", type=str, default="TI-19")
@click.argument("feature_b", type=str, default="FI-19")
@click.argument("objectives", type=str, default="amine")
@click.argument("dataset", type=str, default="full")
def main(feature_a, feature_b, objectives, dataset):
    lower = -20
    upper = 20
    num_points = 21
    # load data
    if dataset == "baseline":
        df = pd.read_pickle("day_0_long_baseline")
    elif dataset == "full":
        df = pd.read_pickle("20210624_df_cleaned.pkl")
    elif dataset == "long_baseline":
        df = pd.read_pickle("day_0_long_long_baseline")

    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    print("starting run")
    results = run_grid(df, feature_a, feature_b, lower, upper, num_points, objectives)
    dump_pickle(
        results,
        f"{TIMESTR}_{feature_a}_{feature_b}_{lower}_{upper}_{num_points}_{objectives}_{str(dataset)}".replace(
            "/", "*"
        ),
    )


if __name__ == "__main__":
    main()
