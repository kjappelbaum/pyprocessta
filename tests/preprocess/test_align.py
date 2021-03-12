# -*- coding: utf-8 -*-
"""Testing the align module"""
import numpy as np
import pandas as pd

from pyprocessta.preprocess.align import align_two_dfs


def test_align(get_timeseries):
    df = get_timeseries
    aligned = align_two_dfs(df, df)
    assert isinstance(aligned, pd.DataFrame)
    assert len(aligned) == len(df)

    dates_a = pd.date_range("1/1/2021", periods=75)
    regular_a = pd.Series(np.arange(75), index=dates_a).to_frame("a")
    dates_b = pd.date_range("1/1/2021", periods=35)
    regular_b = pd.Series(np.arange(35), index=dates_b).to_frame("b")
    aligned = align_two_dfs(regular_a, regular_b)
    assert isinstance(aligned, pd.DataFrame)
    assert len(aligned) == 35
