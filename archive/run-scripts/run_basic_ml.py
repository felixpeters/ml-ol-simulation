import time
import os

from mesa.batchrunner import BatchRunnerMP

from utils.runners import get_info, get_tracking_data
from utils.metrics import *
from utils.analysis import *
from models.basic_ml import BasicMLModel

# collected data will be saved in this folder
DATA_PATH = "data/"
# get the number of available CPUs for multi-processing
CPU_COUNT = os.cpu_count() or 2

# fixed parameters
fixed_params = {
    "belief_dims": 30,
    "num_humans": 50,
    "p_h1": 0.1,
    "p_h2": 0.5,
}

# variable parameters defining each configuration
variable_params = {
    "num_ml": [5, 15],
    "p_1": [0.1, 0.9],
    "p_2": [0.1, 0.9],
    "p_3": [0.1, 0.9],
    "p_ml": [0.2, 0.8],
}

batch_run = BatchRunnerMP(
    BasicMLModel,
    nr_processes=CPU_COUNT,
    variable_parameters=variable_params,
    fixed_parameters=fixed_params,
    iterations=2,
    max_steps=175,
    display_progress=True,
    model_reporters={
        "history": track_model_steps, 
    },
)

# simulation batch run
total_iter, num_conf, num_iter = get_info(batch_run)
print(f'Starting simulation with the following setup:')
print(f'- Total number of iterations: {total_iter}')
print(f'- Number of configurations: {num_conf}')
print(f'- Iterations per configuration: {num_iter}')
print(f'- Number of processing cores: {CPU_COUNT}')
start = time.time()
batch_run.run_all()
end = time.time()
duration = end - start
print(f'Simulation completed after {duration:.2f} seconds (speed: {total_iter/duration:.2f} iterations/second)')

# tracking data
df = get_tracking_data(batch_run)
print(f'Created dataframe from batch run data')
timestr = time.strftime("%Y%m%d-%H%M%S")
print(f'Created raw dataframe with following shape: {df.shape[0]} rows, {df.shape[1]} columns')

# data preprocessing
run_aggs = {
    "belief_dims": "mean",
    "num_humans": "mean",
    "num_ml": "mean",
    "p_1": "mean",
    "p_2": "mean",
    "p_3": "mean",
    "p_h1": "mean",
    "p_h2": "mean",
    "p_ml": "mean",
    "code_kl": ["mean", "std"],
    "human_kl": ["mean", "std"],
    "human_kl_var": "mean",
    "human_kl_dissim": "mean",
}
time_aggs = {
    "belief_dims": "mean",
    "num_humans": "last",
    "num_ml": "last",
    "p_1": "mean",
    "p_2": "mean",
    "p_3": "mean",
    "p_h1": "mean",
    "p_h2": "mean",
    "p_ml": "mean",
    "code_kl": ["max", "last"],
    "human_kl": ["max", "last"],
    "human_kl_var": ["max", "last"],
    "code_kl_std": "last",
    "human_kl_std": "last",
    "human_kl_dissim": ["max", "last"],
}
col_names = {
    "belief_dims_mean": "belief_dims",
    "num_humans_last": "num_humans",
    "num_humans_mean": "num_humans",
    "num_ml_last": "num_ml",
    "num_ml_mean": "num_ml",
    "p_1_mean": "p_1",
    "p_2_mean": "p_2",
    "p_3_mean": "p_3",
    "p_h1_mean": "p_h1",
    "p_h2_mean": "p_h2",
    "p_ml_mean": "p_ml",
    "code_kl_mean": "code_kl",
    "human_kl_mean": "human_kl",
    "human_kl_var_mean": "human_kl_var",
    "human_kl_dissim_mean": "human_kl_dissim",
    "code_kl_std_last": "code_kl_std",
    "human_kl_std_last": "human_kl_std",
}
time_data, agg_data = preprocess_dataset(df, run_aggs, time_aggs, col_names)
time_fname = f"{DATA_PATH}basic_ml_ts_{timestr}.csv"
time_data.to_csv(time_fname)
print(f'Saved time-series dataframe ({time_data.shape[0]} rows, {time_data.shape[1]} columns) to file {time_fname}')
agg_fname = f"{DATA_PATH}basic_ml_agg_{timestr}.csv"
agg_data.to_csv(agg_fname)
print(f'Saved aggregated dataframe ({agg_data.shape[0]} rows, {agg_data.shape[1]} columns) to file {agg_fname}')

# analyze created data
plot_time_series(time_data.reset_index(), 'code_kl', fname=f"analysis_ts_{timestr}")
print('Saved time series plots for code knowledge')
plot_time_series(time_data.reset_index(), 'human_kl_var', fname=f"analysis_ts_{timestr}")
print('Saved time series plots for knowledge variance')
plot_time_series(time_data.reset_index(), 'human_kl_dissim', fname=f"analysis_ts_{timestr}")
print('Saved time series plots for knowledge dissimilarity')
plot_summary(agg_data.reset_index(), 'code_kl_last', fname=f"analysis_agg_{timestr}")
print('Saved summary plots for code knowledge')
