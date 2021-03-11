import pandas as pd
import numpy as np
from pyprocessta.preprocess.smooth import z_score_filter


def test_z_score_filter():
    dates = pd.date_range("1/1/2021", periods=75)
    regular = pd.Series(np.array([1] * 75), index=dates)
    filtered = z_score_filter(regular)
    assert (filtered.values == regular.values).all()
