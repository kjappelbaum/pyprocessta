from pyprocessta.eda.statistics import check_stationarity
import pandas as pd
import numpy as np


def test_check_stationarity():

    dates = pd.date_range("1/1/2021", periods=75)
    regular = pd.Series(np.arange(75) + np.random.normal(0, 0.5, 75), index=dates)
    check_result = check_stationarity(regular)
    assert isinstance(check_result, dict)
    assert not check_result["stationary"]
    assert "adf" in check_result.keys()
    assert "kpss" in check_result.keys()

    dates = pd.date_range("1/1/2021", periods=75)
    regular = pd.Series(np.array([1] * 75) + np.random.normal(0, 0.5, 75), index=dates)
    check_result = check_stationarity(regular)
    assert isinstance(check_result, dict)
    assert check_result["stationary"]
    assert "adf" in check_result.keys()
    assert "kpss" in check_result.keys()
