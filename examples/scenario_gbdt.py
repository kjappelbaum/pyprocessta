# -*- coding: utf-8 -*-
import concurrent.futures
import os
import pickle
from collections import defaultdict
from copy import deepcopy

import click
import joblib
import numpy as np
import pandas as pd

DF = pd.read_pickle("20210508_df_for_scenarios.pkl")


def dump_pickle(obj, filename):
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle)


def run_targets(model, feature_levels):
    df_ = deepcopy(DF)
    for k, v in feature_levels.items():
        if k == "valve-position-12":
            df_[k] = v
        else:
            df_[k] = df_[k] + v / 100 * np.abs(df_[k])

    predictions = model.predict(df_)
    return predictions


def run_grid(
    model,
    feature_a: str = "TI-19",
    feature_b: str = "FI-19",
    lower: float = -20,
    upper: float = 20,
    num_points: int = 21,
    outdir=None,
):
    print(f"Running scenario for {feature_a} and {feature_b}")
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
            results_double_new[point_a][point_b] = run_targets(
                model, {feature_a: point_a, feature_b: point_b}
            )

    if outdir is not None:
        filename = f"{feature_a}_{feature_b}.pkl".replace("/", "*")
        dump_pickle(results_double_new, os.path.join(outdir, filename))

    return results_double_new


FEATURES = [
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
    "deltat",
]


@click.command("cli")
@click.argument("target")
def main(target):
    if target == "co2":
        model = joblib.load("xgb_co2.joblib")
    elif target == "2amp":
        model = joblib.load("xgb_2amp.joblib")
    elif target == "piperazine":
        model = joblib.load("xgb_piperazine.joblib")

    def scenario(feature_tuple):
        return run_grid(
            model=model,
            feature_a=feature_tuple[0],
            feature_b=feature_tuple[1],
            outdir=f"scenario_gbdt_{target}",
        )

    grid = []

    for i, feati in enumerate(FEATURES):
        for j, featj in enumerate(FEATURES):
            if i < j:
                grid.append((feati, featj))

    print("made grid")

    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as exec:
    for _ in map(scenario, grid):
        pass


if __name__ == "__main__":
    main()
