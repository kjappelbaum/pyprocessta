# -*- coding: utf-8 -*-
"""Testing the resample module"""
import pandas as pd

from pyprocessta.preprocess.resample import resample_regular


def test_resample(get_timeseries):
    df = get_timeseries
    resampled = resample_regular(df)
    assert isinstance(resampled, pd.DataFrame)
    assert len(resampled) < len(df)
