# -*- coding: utf-8 -*-
from functools import partial

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

MEAS_COLUMNS = [
    "TI-19",
    "FI-16",
    "TI-33",
    "FI-2",
    "FI-151",
    "TI-8",
    "FI-241",
    "valve-position-12",  # dry-bed
    #     "FI-38",  # strippera
    #     "PI-28",  # stripper
    #     "TI-28",  # stripper
    "FI-20",
    "FI-30",
    "TI-3",
    "FI-19",
    "FI-211",
    "FI-11",
    "TI-30",
    "PI-30",
    "TI-1213",
    #     "TI-4",
    "FI-23",
    "delta_t",
]

TARGETS_clean = [
    "2-Amino-2-methylpropanol C4H11NO",
    "Piperazine C4H10N2",
    "Carbon dioxide CO2",
    "Ammonia NH3",
]


df = pd.read_pickle("df_dropped.pkl")

X, y = df[MEAS_COLUMNS], df[TARGETS_clean].values

scaler = StandardScaler()

X_ = scaler.fit_transform(X)

config = {
    "n_estimators": {"distribution": "int_uniform", "min": 10, "max": 1000},
    "max_depth": {"distribution": "int_uniform", "min": 5, "max": 100},
    "num_leaves": {"distribution": "int_uniform", "min": 5, "max": 500},
    "reg_alpha": {"distribution": "log_uniform", "min": 0.00001, "max": 0.4},
    "reg_lambda": {"distribution": "log_uniform", "min": 0.00001, "max": 0.4},
    "subsample": {"distribution": "uniform", "min": 0.4, "max": 1.0},
    "colsample_bytree": {"distribution": "uniform", "min": 0.4, "max": 1.0},
    "min_child_weight": {
        "distribution": "uniform",
        "min": 0.0001,
        "max": 0.1,
    },
}


def get_sweep_id(method):
    """return us a sweep id (required for running the sweep)"""
    sweep_config = {
        "method": method,
        "metric": {"name": "cv_mean", "goal": "minimize"},
        "early_terminate": {"type": "hyperband", "s": 2, "eta": 3, "max_iter": 30},
        "parameters": config,
    }
    sweep_id = wandb.sweep(sweep_config, project="process_ml")

    return sweep_id


def train(index):
    # Config is a variable that holds and saves hyperparameters and inputs

    configs = {
        "n_estimators": 100,
        "max_depth": 10,
        "num_leaves": 50,
        "reg_alpha": 0.00001,
        "reg_lambda": 0.00001,
        "subsample": 0.2,
        "colsample_bytree": 0.2,
        "min_child_weight": 0.001,
    }

    # Initilize a new wandb run
    wandb.init(project="process_ml", config=configs)

    config = wandb.config
    # config['objective'] =  'huber'

    regressor = LGBMRegressor(**config)

    cv = cross_val_score(
        regressor,
        X_,
        y[:, index],
        n_jobs=-1,
        cv=KFold(n_splits=5),
        scoring="neg_mean_absolute_error",
    )

    mean = np.abs(cv.mean())
    std = np.abs(cv.std())
    wandb.log({"cv_mean": mean})
    wandb.log({"cv_std": std})

    wandb.run.summary["cv_mean"] = mean
    wandb.run.summary["cv_std"] = std


@click.command("cli")
@click.argument("index", type=int)
def main(index):
    sweep_id = get_sweep_id("bayes")
    train_func = partial(train, index=int(index))
    wandb.agent(sweep_id, function=train_func)


if __name__ == "__main__":
    main()
