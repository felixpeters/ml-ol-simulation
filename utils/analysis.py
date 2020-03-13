import pandas as pd

def preprocess_dataset(data, run_aggs, time_aggs, col_names):
    # round values to enable secure indexing
    data = data.round(4)
    data.reset_index(inplace=True)
    # reindex
    configs = pd.unique(data.config)
    runs = pd.unique(data.run)
    steps = pd.unique(data.step)
    index = pd.MultiIndex.from_product([configs, runs, steps], names=['config', 'run', 'step'])
    data.index = index
    data = data.drop(columns=['config', 'run', 'step'])
    # aggregate over runs
    time_data = data.groupby(level=[0, 2]).agg(run_aggs)
    time_data = time_data.round(4)
    time_data.columns = ['_'.join(col).strip() for col in time_data.columns.values]
    time_data.rename(columns=col_names, inplace=True)
    # aggregate over time steps
    agg_data = time_data.groupby(level=0).agg(time_aggs)
    agg_data = agg_data.round(4)
    agg_data.columns = ['_'.join(col).strip() for col in agg_data.columns.values]
    agg_data.rename(columns=col_names, inplace=True)
    return time_data, agg_data
