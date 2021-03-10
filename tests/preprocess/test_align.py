import pandas as pd
from pyprocessta.preprocess.align import align_two_dfs


def test_align(get_timeseries):
    df = get_timeseries
    aligned = align_two_dfs(df, df)
    assert isinstance(aligned, pd.DataFrame)
    assert len(aligned) == len(df)
