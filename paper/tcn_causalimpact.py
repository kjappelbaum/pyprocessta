# -*- coding: utf-8 -*-
import sys

sys.path.append("../src")
import pickle
import time
from copy import deepcopy

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from parse_cesar1 import get_timestep_tuples

from pyprocessta.causalimpact import _select_unrelated_x
from pyprocessta.model.tcn import TCNModelDropout, parallelized_inference

TIMESTR = time.strftime("%Y%m%d-%H%M%S")

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
    # "FI-23",
    # "FI-20",
    # "FI-20/FI-23",
    #    "TI-22",
    "delta_t",
    "TI-35",
    "delta_t_2",
]
HORIZON = 2

df = pd.read_pickle("20210508_df_for_causalimpact.pkl")

with open("step_times.pkl", "rb") as handle:
    times = pickle.load(handle)


def get_causalimpact_splits(x, y, day, times, df):
    a, b = get_timestep_tuples(df, times, day)

    x_way_before, x_after = x.split_before(b[1])
    y_way_before, y_after = y.split_before(pd.Timestamp(b[1]))

    _, x_before_ = x_way_before.split_before(a[0])
    _, y_before_ = y_way_before.split_before(a[0])

    x_before, x_after = x_before_.split_before(a[1])
    y_before, y_after = y_before_.split_before(a[1])

    x_during, x_test = x_after.split_before(b[0])
    y_during, y_test = y_after.split_before(b[0])

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
    ["FI-11", "FI-2"],
    ["TI-1213"],
    ["TI-1213", "TI-19"],
    ["capture rate"],
    # # ["capture rate"],
    # ["valve-position-12"],
]


causalimpact_models = {}


def run_model(x_trains, y_trains):

    # run = wandb.init(project='process_ml', reinit=True, sync_tensorboard=True)
    # with run:

    model_cov = TCNModelDropout(
        input_chunk_length=30,
        output_chunk_length=HORIZON,
        num_layers=4,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
        weight_norm=True,
        batch_size=32,
        n_epochs=200,
        log_tensorboard=True,
        # optimizer_kwargs={"lr": 5e-6},
    )

    model_cov.fit(series=y_trains, covariates=x_trains, verbose=False)

    return model_cov


def transform_data_new(train_tuple, test_tuples, all_tuple):
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

    x_all = transformer.transform(all_tuple[0])
    y_all = transformer.transform(all_tuple[1])
    return (
        (x_train, y_train),
        transformed_test_tuples,
        (transformer, y_transformer),
        (x_all, y_all),
    )


if __name__ == "__main__":
    TARGETS = ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"]
    # TARGETS = ["Carbon dioxide CO2", "Ammonia NH3"]
    for day in range(len(step_changes)):
        cols = deepcopy(MEAS_COLUMNS)
        if step_changes[day][0] in MEAS_COLUMNS:
            for var in step_changes[day]:
                try:
                    cols = _select_unrelated_x(df, cols, var, 0.01)
                    print(TARGETS, day, var, cols)
                    y = TimeSeries.from_dataframe(df[TARGETS])
                    x = TimeSeries.from_dataframe(df[cols])

                    x = Scaler().fit_transform(x)
                    y = Scaler().fit_transform(y)

                    x_trains = []
                    y_trains = []

                    # for day_ in range(len(step_changes)):
                    #     before, during, after = get_causalimpact_splits(
                    #         x, y, day_, times, df
                    #     )

                    #     x_trains.append(before[0])
                    #     y_trains.append(before[1])

                    before, during, after = get_causalimpact_splits(
                        x, y, day, times, df
                    )

                    x_trains.append(before[0])
                    y_trains.append(before[1])

                    before_x_df, before_y_df = (
                        before[0].pd_dataframe(),
                        before[1].pd_dataframe(),
                    )
                    during_x_df, during_y_df = (
                        during[0].pd_dataframe(),
                        during[1].pd_dataframe(),
                    )
                    after_x_df, after_y_df = (
                        after[0].pd_dataframe(),
                        after[1].pd_dataframe(),
                    )

                    day_x_df = pd.concat([before_x_df, during_x_df, after_x_df], axis=0)
                    day_x_ts = TimeSeries.from_dataframe(day_x_df)

                    day_y_df = pd.concat([before_y_df, during_y_df, after_y_df], axis=0)
                    day_y_ts = TimeSeries.from_dataframe(day_y_df)

                    model = run_model(x_trains, y_trains)

                    causalimpact_models[day] = {
                        "before": before,
                        "during": during,
                        "after": after,
                        "model": model,
                        "x_all": day_x_ts,
                        "y_all": day_y_ts,
                        "cols": cols,
                    }
                except Exception as e:
                    print(e)
                    pass

    for day, values in causalimpact_models.items():
        if step_changes[day][0] in MEAS_COLUMNS:
            try:
                model = values["model"]
                predictions = parallelized_inference(
                    model,
                    values["x_all"],
                    values["y_all"],
                    start=(len(values["before"][0]) - 0.4 * len(values["before"][0]))
                    / len(values["y_all"]),
                    repeats=10,
                    horizon=HORIZON,
                )

                causalimpact_models[day]["predictions"] = predictions
            except Exception as e:
                print(e)
                pass

        with open(
            f"{TIMESTR}-_{day}_causalimpact_{'_'.join(TARGETS)}.pkl".replace("/", "*"),
            "wb",
        ) as handle:
            pickle.dump(causalimpact_models, handle)
