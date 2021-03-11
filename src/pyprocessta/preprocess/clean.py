import pandas as pd
from typing import Union


def drop_duplicated_indices(
    df: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """If one concatenates dataframes there might be duplicated 
    indices. This can lead to problems, e.g., in interpolation steps.
    One easy solution can be to just drop the duplicated row

    Args:
        df (Union[pd.Series, pd.DataFrame]): Input data

    Returns:
        Union[pd.Series, pd.DataFrame]: Data without duplicated indices
    """
    return df[~df.index.duplicated()]
