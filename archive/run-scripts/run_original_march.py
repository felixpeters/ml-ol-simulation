import time
import os

from mesa.batchrunner import BatchRunnerMP

from utils.runners import get_info, get_tracking_data
from utils.metrics import *
from utils.analysis import preprocess_dataset
from models.original_march import OriginalMarchModel

# collected data will be saved in this folder
DATA_PATH = "data/"
# get the number of available CPUs for multi-processing
CPU_COUNT = os.cpu_count() or 2

# fixed parameters
fixed_params = {
    "belief_dims": 30,
    "num_humans": 50,
}

# variable parameters defining each configuration
variable_params = {
    "p_1": [0.1, 0.5, 0.9],
    "p_2": [0.1, 0.5, 0.9],
}

batch_run = BatchRunnerMP(
    OriginalMarchModel,
    nr_processes=CPU_COUNT,
    variable_parameters=variable_params,
    fixed_parameters=fixed_params,
    iterations=80,
    max_steps=100,
    display_progress=True,
    model_reporters={
        "history": track_model_steps, 
        "ACK": calc_code_kl, 
        "AHK": calc_human_kl, 
        "VHK": calc_kl_var,
        "DISSIM": calc_dissim,
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
fname = f"{DATA_PATH}original_march_raw_{timestr}.csv"
df.to_csv(fname)
print(f'Saved raw dataframe ({df.shape[0]} rows, {df.shape[1]} columns) to file {fname}')

# data preprocessing
time_aggs = {
    "belief_dims": "mean",
    "num_humans": "last",
    "p_1": "mean",
    "p_2": "mean",
    "code_kl": ["max", "last"],
    "human_kl": ["max", "last"],
    "human_kl_var": ["max", "last"],
    "human_kl_dissim": ["max", "last"],
}
col_names = {
    "belief_dims_mean": "belief_dims",
    "num_humans_last": "num_humans",
    "p_1_mean": "p_1",
    "p_2_mean": "p_2",
}
time_data, agg_data = preprocess_dataset(fname, time_aggs, col_names)
time_fname = f"{DATA_PATH}original_march_ts_{timestr}.csv"
time_data.to_csv(time_fname)
print(f'Saved time-series dataframe ({time_data.shape[0]} rows, {time_data.shape[1]} columns) to file {time_fname}')
agg_fname = f"{DATA_PATH}original_march_agg_{timestr}.csv"
agg_data.to_csv(agg_fname)
print(f'Saved aggregated dataframe ({agg_data.shape[0]} rows, {agg_data.shape[1]} columns) to file {agg_fname}')
