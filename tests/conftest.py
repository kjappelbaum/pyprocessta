import pytest
import pandas as pd
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture(scope='module')
def get_timeseries():
    return pd.read_pickle(os.path.join(THIS_DIR, "data", "example_df.pkl"))

