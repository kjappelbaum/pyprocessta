import pickle
import pandas as pd
from typing import Union


def dump_as_pickle(obj, filename):
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle)


def is_regular_grid(series: Union[pd.Series, pd.DatetimeIndex]) -> bool:
    """For many analyses it can be convenient to have
    the data on a regular grid. This function checks 
    if this is the case. 

    Args:
        series (pd.Series): pd.Series of datetime

    Returns:
        bool: [description]
    """
    if isinstance(series, pd.DatetimeIndex):
        series = pd.Series(series, series)
    return len(series.diff().dropna().unique()) == 1
