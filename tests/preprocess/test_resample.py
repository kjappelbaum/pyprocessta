from pyprocessta.preprocess.resample import resample_regular
import pandas as pd


def test_resample(get_timeseries):
    df = get_timeseries
    resampled = resample_regular(df)
    assert isinstance(resampled, pd.DataFrame)
    assert len(resampled) < len(df)

