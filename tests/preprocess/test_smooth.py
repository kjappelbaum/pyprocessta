# -*- coding: utf-8 -*-
"""Testing the smooth module"""
import numpy as np
import pandas as pd

from pyprocessta.preprocess.smooth import (exponential_window_smoothing,
                                           z_score_filter)


def test_z_score_filter():
    dates = pd.date_range("1/1/2021", periods=75)
    regular = pd.Series(np.array([1] * 75), index=dates)
    filtered = z_score_filter(regular)
    assert (filtered.values == regular.values).all()


def test_exponential_window_smoothing():
    dates = pd.date_range("1/1/2021", periods=75)
    regular = pd.Series(np.array([1] * 75), index=dates)
    filtered = exponential_window_smoothing(regular, 10)
    assert len(filtered) == 75
