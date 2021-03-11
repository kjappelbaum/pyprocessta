from typing import Union
import pandas as pd
import numpy as np
from scipy import stats
from copy import deepcopy


def exponential_window_smoothing(
    data: Union[pd.Series, pd.DataFrame], window_size: int, aggregation: str = "mean"
) -> Union[pd.Series, pd.DataFrame]:
    """

    Args:
        data (Union[pd.Series, pd.DataFrame]): Data to smoothen
        window_size (int): size for the exponential window
        aggregation (str, optional): Aggregation function. Defaults to "mean".

    Returns:
        Union[pd.Series, pd.DataFrame]: Smoothned data
    """
    if aggregation == "median":
        new_data = data.ewm(span=window_size).median()
    else:
        new_data = data.ewm(span=window_size).mean()
    return new_data


def z_score_filter(
    series: pd.Series, threshold: float = 2, window: int = 10
) -> pd.Series:
    """

    Args:
        series (pd.Series): Series to despike 
        threshold (float, optional): Threshold on the z-score. Defaults to 2.
        window (int, option): Window that is used for the median with which 
            the spike value is replaced. This mean only looks back.

    Returns:
        pd.Series: Despiked series
    """
    indices = series.index
    series_ = deepcopy(series.values)
    mask = np.abs(stats.zscore(series_)) > threshold
    indices_masked = np.where(mask)[0]
    for index in indices_masked:
        series_[index] = np.median(series_[int(index - window) : index-1])
    return pd.Series(series_, index=indices)
