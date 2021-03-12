# -*- coding: utf-8 -*-
"""Sometimes, different kinds of measurements are sampled at different intervals. This module provides utilities to combine such data.
We will always operate on pandas dataframes with datatime indexing
"""
from typing import Union

import pandas as pd

from .resample import _interpolate


def align_two_dfs(
    df_a: pd.DataFrame, df_b: pd.DataFrame, interpolation: Union[str, int] = "linear"
) -> pd.DataFrame:
    """Alignes to dataframes with datatimeindex
    Resamples both dataframes on the dataframe with the lowest frequency timestep.
    The first timepoint in the new dataframe will be the later one of the first
    observations of the dataframes.

    Args:
        df_a (pd.DataFrame): Dataframe
        df_b (pd.DataFrame): Dataframe
        interpolation (Union[str, int], optional): Interpolation method.
            If you provide an integer, spline interpolation of that order will be used.
            Defaults to "linear".

    Returns:
        pd.DataFrame: merged dataframe
    """
    assert isinstance(df_a, pd.DataFrame)
    assert isinstance(df_b, pd.DataFrame)

    index_series_a = pd.Series(df_a.index, df_a.index)
    index_series_b = pd.Series(df_b.index, df_b.index)
    timestep_a = max(index_series_a.diff().dropna())
    timestep_b = max(index_series_b.diff().dropna())

    if timestep_a > timestep_b:
        resample_step = timestep_a
    else:
        resample_step = timestep_b

    start_time = max([df_a.index[0], df_b.index[0]])

    resampled_a = df_a.resample(resample_step, origin=start_time)
    resampled_b = df_b.resample(resample_step, origin=start_time)

    new_df_a = _interpolate(resampled_a, interpolation)
    new_df_b = _interpolate(resampled_b, interpolation)

    merged = pd.merge(new_df_a, new_df_b, left_index=True, right_index=True)

    return merged
