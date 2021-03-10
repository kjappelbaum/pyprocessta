import pandas as pd
from typing import Union


def _interpolate(resampled, interpolation):
    if isinstance(interpolation, int):
        result = resampled.interpolate("spline", interpolation)
    else:
        result = resampled.interpolate("linear")
    return result


def resample_regular(
    df: pd.DataFrame, interval: str = "10min", interpolation: Union[str, int] = "linear"
) -> pd.DataFrame:
    """Resamples the dataframe at a desired interval.

    Args:
        df (pd.DataFrame): input dataframne
        interval (str, optional): Resampling intervall. Defaults to "10min".
        interpolation (Union[str, int], optional): Interpolation method. 
            If you provide an integer, spline interpolation of that order will be used. 
            Defaults to "linear".

    Returns:
        pd.DataFrame: Output data. 
    """
    resampled = df.resample(interval, origin="start")
    result = _interpolate(resampled, interpolation)
    return result

