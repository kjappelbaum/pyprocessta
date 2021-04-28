from collections import defaultdict
import pickle
from pyprocessta.model.tcn import parallelized_inference, summarize_results
import pandas as pd
from darts import TimeSeries
from copy import deepcopy
import numpy as np
import joblib
import click
import time


def load_pickle(filename):
    with open(filename, "rb") as handle:
        res = pickle.load(handle)
    return res


def dump_pickle(object, filename):
    with open(filename, "wb") as handle:
        pickle.dump(object, handle)


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

# First, we train a model on *all* data
# Then, we do a partial-denpendency plot approach and change one variable and see how the model predictions change

# load the trained model

FEAT_NUM_MAPPING = dict(zip(MEAS_COLUMNS, [str(i) for i in range(len(MEAS_COLUMNS))]))
UPDATE_MAPPING = {
    "amine": {
        "scaler": joblib.load("y_transformer_2amp_pip"),
        "model": joblib.load("2amp_pip_model"),
        "name": ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"],
    },
    "co2": {
        "scaler": joblib.load("y_transformer_co2_ammonia"),
        "model": joblib.load("co2_ammonia_model"),
        "name": ["Carbon dioxide CO2", "Ammonia NH3"],
    },
}
SCALER = joblib.load("x_scaler")


# load data
DF = pd.read_pickle("df_dropped_5min_resampled.pkl")


def run_update(x, target="amine"):
    model_dict = UPDATE_MAPPING[target]
    y = model_dict["scaler"].transform(
        TimeSeries.from_dataframe(DF, value_cols=model_dict["name"])
    )
    df = parallelized_inference(model_dict["model"], x, y, repeats=10)
    means, stds = summarize_results(df)
    return {"means": means, "stds": stds}


def run_targets(feature_levels, target: str = "amine"):

    df = deepcopy(DF)

    for k, v in feature_levels.items():
        df[k] = df[k] + v / 100 * np.abs(df[k])

    X_ = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS)
    X_ = SCALER.transform(X_)

    res = run_update(X_, target)
    return res


def run_grid(
    feature_a: str = "TI-19",
    feature_b: str = "FI-19",
    lower: float = -20,
    upper: float = 20,
    num_points: int = 21,
    objectives: str = "amine",
):
    grid = np.linspace(lower, upper, num_points)
    results_double_new = defaultdict(dict)
    for point_a in grid:
        for point_b in grid:
            results_double_new[point_a][point_b] = run_targets(
                {feature_a: point_a, feature_b: point_b}, objectives
            )
    return results_double_new


@click.command("cli")
@click.argument("feature_a", type=str, default="TI-19")
@click.argument("feature_b", type=str, default="FI-19")
@click.argument("lower", type=float, default=-20)
@click.argument("upper", type=float, default=20)
@click.argument("num_points", type=int, default=21)
@click.argument("objectives", type=str, default="amine")
def main(feature_a, feature_b, lower, upper, num_points, objectives):
    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    print("starting run")
    results = run_grid(feature_a, feature_b, lower, upper, num_points, objectives)
    dump_pickle(
        results,
        f"{TIMESTR}_{feature_a}_{feature_b}_{lower}_{upper}_{num_points}_{objectives}",
    )


if __name__ == "__main__":
    main()
