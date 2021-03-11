import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np


def check_stationarity(
    series: pd.Series, threshold: float = 0.05, regression="c"
) -> dict:
    """Performs the Augmented-Dickey fuller and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests
    for stationarity.

    Args:
        series (pd.Series): Time series data
        threshold (float, optional): p-value thresholds for the statistical tests. 
            Defaults to 0.05.
        regression (str, optional): If regression="c" then the tests check for stationarity around a constant.
            For "ct" the test check for stationarity around a trend.
            Defaults to "c".

    Returns:
        dict: Results dictionary with key "stationary" that has a bool as value
    """

    assert regression in ["c", "ct"]

    adf_results = adfuller(series, regression=regression)
    kpss_results = kpss(series, regression=regression, nlags="auto")

    # null hypothesis for ADF is non-sationarity for KPSS null hypothesis is stationarity
    conclusion = (kpss_results[1] > threshold) & (adf_results[1] < threshold)
    results = {
        "adf": {
            "statistic": adf_results[0],
            "p_value": adf_results[1],
            "stationary": adf_results[1] < threshold,
        },
        "kpss": {
            "statistic": kpss_results[0],
            "p_value": kpss_results[1],
            "stationary": kpss_results[1] > threshold,
        },
        "stationary": conclusion,
    }

    return results


def check_granger_causality(
    x: pd.Series, y: pd.Series, max_lag: int = 20, add_constant: bool = True
) -> dict:
    test_result = grangercausalitytests(
        np.hstack([x.values, y.values]),
        max_lag=max_lag,
        add_constant=add_constant,
        verbose=False,
    )

    return test_result


def computer_granger_causality_matrix(df: pd.DataFrame) -> pd.DataFrame:
    ...
