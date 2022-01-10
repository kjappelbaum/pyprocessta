import wandb
from darts.models import TCNModel
from pyprocessta.model.utils import split_data
import pandas as pd
from darts.metrics import mape, mae
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

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

TARGETS_clean = ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"]

sweep_config = {
    "name": "",
    "method": "bayes",
    "parameters": {
        "num_layers": {"values": [2, 4, 8, 16]},
        "num_filters": {"values": [8, 16, 32, 64]},
        "weight_norm": {"values": [True, False]},
        "kernel_size": {"values": [2, 3, 4, 5]},
        "dropout": {
            "min": 0.1,
            "max": 0.9,
        },
        "batch_size": [32, 64, 128],
        "num_outputs": {"values": [0, 1]},
        "n_epochs": {"values": [100, 200, 300, 400]},
        "input_chunk_length": {"values": [30, 40, 60, 80, 160]},
        "lr": {"min": -5, "max": -1, "distribution": "log_uniform"},
    },
}

sweep_id = wandb.sweep(sweep_config)


df = pd.read_pickle("../paper/20210624_df_cleaned.pkl")
Y = TimeSeries.from_dataframe(df, value_cols=TARGETS_clean)
X = TimeSeries.from_dataframe(df, value_cols=MEAS_COLUMNS)


def get_data(num_outputs):
    x = X
    y = Y if num_outputs == 2 else Y[TARGETS_clean[0]]
    train, valid, test = split_data(x, y, 0.5)

    return train, valid, test


def train_test(config):
    run = wandb.init(config)

    train, valid, _ = get_data(run.config.num_outputs)
    model_cov = TCNModel(
        input_chunk_length=run.config.input_chunk_length,
        output_chunk_length=30,
        num_layers=run.config.num_layers,
        num_filters=run.config.num_filters,
        kernel_size=run.config.kernel_size,
        dropout=run.config.dropout,
        weight_norm=run.config.weight_norm,
        batch_size=run.config.batch_size,
        n_epochs=run.config.n_epochs,
        log_tensorboard=False,
        optimizer_kwargs={"lr": run.config.lr},
    )

    model_cov.fit(series=train[1], past_covariates=train[0], verbose=False)

    backtest_train = model_cov.historical_forecasts(
        train[1],
        past_covariates=train[0],
        start=0.1,
        forecast_horizon=30,
        stride=1,
        retrain=False,
        verbose=False,
    )

    backtest_valid = model_cov.historical_forecasts(
        valid[1],
        past_covariates=valid[0],
        start=0.1,
        forecast_horizon=30,
        stride=1,
        retrain=False,
        verbose=False,
    )

    mape_valid = mape(valid[TARGETS_clean[0]], backtest_valid["0"])
    mape_train = mape(train[TARGETS_clean[0]], backtest_train["0"])

    mae_valid = mae(valid[TARGETS_clean[0]], backtest_valid["0"])
    mae_train = mae(train[TARGETS_clean[0]], backtest_train["0"])

    wandb.log({"mape_valid": mape_valid})
    wandb.log({"mape_train": mape_train})

    wandb.log({"mae_valid": mae_valid})
    wandb.log({"mae_train": mae_train})


if __name__ == "__main__":
    wandb.agent(sweep_id, train_test)
