import datetime
from collections import Counter
from itertools import product, count
from functools import reduce

from tqdm import tqdm
import pandas as pd
from mesa.time import BaseScheduler
from mesa.batchrunner import BatchRunner

def get_info(runner):
    num_configs = reduce(lambda prod, params: prod * len(params), runner.variable_parameters.values(), 1)
    num_runs = runner.iterations
    return (num_configs * num_runs, num_configs, num_runs)

def get_tracking_data(runner):
    # get number of configurations, runs and steps
    num_configs = reduce(lambda prod, params: prod * len(params), runner.variable_parameters.values(), 1)
    num_runs = runner.iterations
    num_steps = runner.max_steps
    # create MultiIndex from cross product
    configs = list(range(1, num_configs+1))
    runs = list(range(1, num_runs+1))
    steps = list(range(0, num_steps+1))
    index = pd.MultiIndex.from_product([configs, runs, steps], names=['config', 'run', 'step'])
    # assemble data frame from model tracking data
    df = runner.get_model_vars_dataframe()
    hists = df.loc[:,'history']
    res_df = pd.DataFrame()
    for hist in hists:
        hist_df = pd.DataFrame(hist.model_vars)
        hist_df = hist_df.drop(columns=["time"])
        res_df = res_df.append(hist_df)
    # reset index to created MultiIndex
    res_df.index = index
    return res_df