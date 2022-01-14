# -*- coding: utf-8 -*-
"""
Causal impact analysis uses machine learning to construct a counterfactual
(what would the results have been without an intervention)
which can be used to estimate an effect size without the need for a control group
A good introduction is
https://www.youtube.com/watch?v=GTgZfCltMm8
the original paper is
https://storage.googleapis.com/pub-tools-public-publication-data/pdf/41854.pdf.
The particular implementation we use was described in
https://towardsdatascience.com/implementing-causal-impact-on-top-of-tensorflow-probability-c837ea18b126

One can use covariates to build the counterfactual but one needs to be careful that
they are not changed by the intervention.
"""
from typing import List

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from causalimpact import CausalImpact
from causalimpact.misc import standardize
from causalimpact.model import build_bijector, build_inv_gamma_sd_prior

from ..eda.statistics import check_granger_causality

__all__ = ["run_causal_impact_analysis"]


def _select_unrelated_x(df, x_columns, intervention_column, p_value_threshold, lag=10):
    unrelated_x = []

    for x_column in x_columns:
        if x_column != intervention_column:
            granger_result = check_granger_causality(
                df[x_column], df[intervention_column], lag
            )
            if granger_result["min_p_value"] > p_value_threshold:
                unrelated_x.append(x_column)

    return unrelated_x


def run_causal_impact_analysis(
    df: pd.DataFrame,
    x_columns: List[str],
    intervention_column: str,
    y_column: str,
    start: List,
    end: List,
    p_value_threshold: float = 0.05,
) -> object:
    """Run the causal impact analysis.
    Here, we use all the x that are not related
    to the intervention variable.

    Args:
        df (pd.DataFrame): Dataframe to run the analysis on
        x_columns (List[str]): All column names that can
            potentially be used as covariates for
            the counterfactual model
        intervention_column (str): Name of the column
            on which the intervention has been performed
        y_column (str): Target column on which we want
            to understand the effect of the intervention
        start (List): Two elements defining the pre-intervention
            interval
        end (List): Two elements defining the post-intervention
            interval
        p_value_threshold (float): H0 that x does not Granger cause
            y is rejected when p smaller this threshold. Defaults to 0.05.

    Returns:
        object: causalimpact object
    """
    # First, we need to remove all the columns from X that are somehow related to our intervention variable
    # this avoids that we bias the estimation of our causal effect by data leakage
    x_columns = _select_unrelated_x(
        df, x_columns, intervention_column, p_value_threshold
    )
    new_data = df[[y_column] + x_columns]

    # model = _linear_trend_linear_reg_model(new_data, start, end)

    # now we can run the causal impact analysis
    ci = CausalImpact(
        new_data,
        start,
        end,
        # model=model,
        model_args={"standardize": True, "nseasons": 16, "prior_sd": 0.01},
    )

    return ci


def _linear_trend_linear_reg_model(df, start, end):
    df = df.astype(np.float32)
    normed_data, _ = standardize(df)
    obs_data = df.loc[start[0] : start[1]].iloc[:, 0]
    obs_sd = np.sqrt(obs_data.std())
    sd_prior = build_inv_gamma_sd_prior(0.01)
    sd_prior = build_bijector(sd_prior)
    # This is an approximation to simulate the bsts package from R. It's expected that
    # given a few data points the posterior will converge appropriately given this
    # distribution, that's why it's divided by 2.
    obs_prior = build_inv_gamma_sd_prior(obs_sd / 2)
    obs_prior = build_bijector(obs_prior)

    local = tfp.sts.SemiLocalLinearTrend(
        observed_time_series=obs_data, level_scale_prior=sd_prior
    )
    linear_reg = tfp.sts.SparseLinearRegression(
        design_matrix=normed_data.loc[start[0] : end[1]]
        .iloc[:, 1:]
        .values.reshape(-1, normed_data.shape[1] - 1)
    )

    model = tfp.sts.Sum(
        [local, linear_reg],
        observed_time_series=obs_data,
        observation_noise_scale_prior=obs_prior,
    )

    return model
