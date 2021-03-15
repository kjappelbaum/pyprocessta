# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pyprocessta.causalimpact import (_select_unrelated_x,
                                      run_causal_impact_analysis)


def test_remove_unrelated_columns():
    # Check if we correctly remove the unrelated columns
    dates = pd.date_range("1/1/2021", periods=75)
    a = pd.Series(
        np.array([x + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="a",
    )
    b = pd.Series(
        np.array([x + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="b",
    )
    c = pd.Series(
        np.array([x + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="c",
    )

    d = pd.Series(
        np.array([1 + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="d",
    )

    data = pd.concat([a, b, c, d], axis=1)
    selected_columns = _select_unrelated_x(data, ["a", "b", "c", "d"], "a", 0.01, 1)
    assert selected_columns == ["d"]


def test_causalimpact_analysis():
    dates = pd.date_range("1/1/2021", periods=75)
    a = pd.Series(
        np.array([x + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="a",
    )
    b = pd.Series(
        np.array([x + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="b",
    )
    c = pd.Series(
        np.array([x + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="c",
    )

    d = pd.Series(
        np.array([1 + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="d",
    )

    e = pd.Series(
        np.array([4 + np.random.normal(0, 0.05) for x in range(75)]),
        index=dates,
        name="e",
    )
    data = pd.concat([a, b, c, d, e], axis=1)

    s_0 = dates[0]
    s_1 = dates[10]
    e_0 = dates[11]
    e_1 = dates[20]

    ci = run_causal_impact_analysis(
        df=data,
        x_columns=["a", "b", "c"],
        intervention_column="a",
        y_column="e",
        start=[s_0, s_1],
        end=[e_0, e_1],
    )

    assert isinstance(ci, object)
