import sys

sys.path.append("../src")
from parse_cesar1 import get_timestep_tuples
import pickle
from pyprocessta.causalimpact import _select_unrelated_x
from pyprocessta.model.tcn import transform_data, run_model, parallelized_inference

from darts import TimeSeries
import pandas as pd
from copy import deepcopy
import time

TIMESTR = time.strftime("%Y%m%d-%H%M%S")

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

df = pd.read_pickle("detrended.pkl")
df = df.dropna()
df["delta_t"] = df["TI-35"] - df["TI-4"]
MEAS_COLUMNS.append("delta_t")

with open("step_times.pkl", "rb") as handle:
    times = pickle.load(handle)


def get_causalimpact_splits(x, y, day, times, df):
    a, b = get_timestep_tuples(df, times, day)

    x_way_before, x_after = x.split_before(a[1])
    y_way_before, y_after = y.split_before(pd.Timestamp(a[1]))

    _, x_before = x_way_before.split_before(a[0])
    _, y_before = y_way_before.split_before(a[0])

    x_during, x_test = x_after.split_before(b[1])
    y_during, y_test = y_after.split_before(b[1])

    return (x_before, y_before), (x_during, y_during), (x_test, y_test)


# The experiments with the faster response on emissions are:

# - Day 1: water wash temperature step increase. 6~min time delay in amine and CO2 emissions.
# - Day 5: lean solvent and flue gas flow step decrease. 6~min time delay in amine emissions.
# - Day 7: lean solvent and water wash temperature step decrease. 6 min time delay in PZ emissions and 12~min delay in the response of AMP emissions.
# - Day 8: capture rate step decrease. And consequently reboiler level step decrease.
# - Day 9: dry bed operation


# If the time delay and the magnitude of emissions are both taken into consideration, then the experiments with the highest effect on emissions (ranking the one with the highest effect starting from the top to the bottom) become:

# - Day 9: dry bed operation
# - Day 7: lean solvent and water wash temperature step decrease
# - Day 1: water wash temperature step increase

step_changes = [
    ["TI-19"],
    ["FI-19"],
    ["TI-3"],
    ["FI-11"],
    ["FI-11, FI-3"],
    ["TI-1213"],
    ["TI-1213", "TI-19"],
    ["capture rate"],
    ["capture rate"],
    ["valve-position-12"],
]


causalimpact_models = {}


if __name__ == "__main__":
    TARGETS = ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"]
    for day in range(10):
        cols = deepcopy(MEAS_COLUMNS)
        if step_changes[day][0] in MEAS_COLUMNS:
            for var in step_changes[day]:
                try:
                    cols = _select_unrelated_x(df, cols, var, 0.05)
                    y = TimeSeries.from_dataframe(df, value_cols=TARGETS)
                    x = TimeSeries.from_dataframe(df, value_cols=cols)

                    before, during, after = get_causalimpact_splits(
                        x, y, day, times, df
                    )
                    train_tuple, test_tuple, scalers = transform_data(
                        before, [during, after]
                    )
                    model = run_model(train_tuple)

                    causalimpact_models[day] = {
                        "before": before,
                        "during": during,
                        "after": after,
                        "model": model,
                    }
                except Exception:
                    pass

    for day, values in causalimpact_models.items():
        if step_changes[day][0] in MEAS_COLUMNS:
            try:
                x_test, y_test = values["before"]
                model = values["model"]
                before_predictions = parallelized_inference(
                    model, x_test, y_test, start=0.2
                )

                x_test, y_test = values["during"]
                during_predictions = parallelized_inference(
                    model, x_test, y_test, start=0.2
                )

                x_test, y_test = values["after"]
                after_predictions = parallelized_inference(
                    model, x_test, y_test, start=0.2
                )

                causalimpact_models[day]["before_predictions"] = before_predictions
                causalimpact_models[day]["during_predictions"] = during_predictions
                causalimpact_models[day]["after_predictions"] = after_predictions
            except Exception as e:
                print(e)
                pass

    with open(f"{TIMESTR}-causalimpact_2amp_pip.pkl", "wb") as handle:
        pickle.dump(causalimpact_models, handle)