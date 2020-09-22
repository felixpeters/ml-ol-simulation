import base64
from io import BytesIO
from itertools import product

import pandas as pd
import numpy as np


run_aggs = {
    "m": "mean",
    "n": "mean",
    "j": "mean",
    "p_1": "mean",
    "p_2": "mean",
    "p_3": "mean",
    "q_h1": "mean",
    "q_h2": "mean",
    "q_ml": "mean",
    "alpha_ml": "mean",
    "p_turb": "mean",
    "q_ml_scaling": "last",
    "avg_q_ml": "mean",
    "code_kl": ["mean", "std"],
    "human_kl": ["mean", "std"],
    "human_kl_var": "mean",
    "human_kl_dissim": "mean",
}
time_aggs = {
    "m": "mean",
    "n": "last",
    "j": "last",
    "p_1": "mean",
    "p_2": "mean",
    "p_3": "mean",
    "q_h1": "mean",
    "q_h2": "mean",
    "q_ml": "mean",
    "alpha_ml": "mean",
    "p_turb": "mean",
    "q_ml_scaling": "last",
    "avg_q_ml": ["max", "last"],
    "code_kl": ["max", "last"],
    "human_kl": ["max", "last"],
    "human_kl_var": ["max", "last"],
    "code_kl_std": "last",
    "human_kl_std": "last",
    "human_kl_dissim": ["max", "last"],
}
col_names = {
    "m_mean": "m",
    "n_last": "n",
    "n_mean": "n",
    "j_last": "j",
    "j_mean": "j",
    "p_1_mean": "p_1",
    "p_2_mean": "p_2",
    "p_3_mean": "p_3",
    "q_h1_mean": "q_h1",
    "q_h2_mean": "q_h2",
    "q_ml_mean": "q_ml",
    "alpha_ml_mean": "alpha_ml",
    "p_turb_mean": "p_turb",
    "avg_q_ml_mean": "avg_q_ml",
    "code_kl_mean": "code_kl",
    "q_ml_scaling_last": "q_ml_scaling",
    "human_kl_mean": "human_kl",
    "human_kl_var_mean": "human_kl_var",
    "human_kl_dissim_mean": "human_kl_dissim",
    "code_kl_std_last": "code_kl_std",
    "human_kl_std_last": "human_kl_std",
}


def preprocess_dataset(data, run_aggs, time_aggs, col_names):
    # round values to enable secure indexing
    data = data.round(4)
    data.reset_index(inplace=True)
    # reindex
    configs = pd.unique(data.config)
    runs = pd.unique(data.run)
    steps = pd.unique(data.step)
    index = pd.MultiIndex.from_product(
        [configs, runs, steps], names=['config', 'run', 'step'])
    data.index = index
    data = data.drop(columns=['config', 'run', 'step'])
    # aggregate over runs
    time_data = data.groupby(level=[0, 2]).agg(run_aggs)
    time_data = time_data.round(4)
    time_data.columns = ['_'.join(col).strip()
                         for col in time_data.columns.values]
    time_data.rename(columns=col_names, inplace=True)
    # aggregate over time steps
    agg_data = time_data.groupby(level=0).agg(time_aggs)
    agg_data = agg_data.round(4)
    agg_data.columns = ['_'.join(col).strip()
                        for col in agg_data.columns.values]
    agg_data.rename(columns=col_names, inplace=True)
    return time_data, agg_data
