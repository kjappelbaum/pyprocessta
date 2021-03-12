# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pyprocessta.utils import is_regular_grid


def test_is_regular_grid():
    dates = pd.date_range("1/1/2021", periods=75)
    regular = pd.Series(np.arange(75), index=dates)
    assert is_regular_grid(regular)
    assert is_regular_grid(dates)

    dates_2 = pd.date_range("1/1/2021", periods=7)
    regular_2 = pd.Series(np.arange(7), index=dates_2)
    irregular = pd.concat([regular, regular_2])
    assert not is_regular_grid(irregular)
    assert not is_regular_grid(irregular.index)
