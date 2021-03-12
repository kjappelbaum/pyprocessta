import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from typing import List
from collections import defaultdict


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
    results = {}
    test_result = grangercausalitytests(
        np.hstack([x.values.reshape(-1, 1), y.values.reshape(-1, 1)]),
        maxlag=max_lag,
        addconst=add_constant,
        verbose=False,
    )
    results["detail"] = test_result
    p_values = []

    for _, v in test_result.items():
        p_values.append(v[0]["ssr_chi2test"][1])

    results["max_p_value"] = max(p_values)
    results["lag_w_max_p_value"] = np.argmax(p_values)
    return results


def computer_granger_causality_matrix(
    df: pd.DataFrame, xs: List[str], ys: List[str]
) -> pd.DataFrame:
    results_matrix = defaultdict(list)

    for x in xs:
        for y in ys:
            results_matrix[x].append(
                check_granger_causality(df[x], df[y])["max_p_value"]
            )

    return pd.DataFrame.from_dict(results_matrix, orient="index", columns=ys)
