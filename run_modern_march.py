import time
import os

from mesa.batchrunner import BatchRunnerMP

from utils.runners import get_info, get_tracking_data
from utils.metrics import track_model_steps, calc_code_kl, calc_human_kl
from models.modern_march import ModernMarchModel

# constants
DATA_PATH = "data/"
CPU_COUNT = os.cpu_count() or 2

# batch run configuration
fixed_params = {
    "belief_dims": 30,
    "num_humans": 50,
}

variable_params = {
    "p_1": [0.2, 0.5, 0.8],
    "p_2": [0.2, 0.5, 0.8],
    "p_hp": [0.05, 0.1],
    "p_hm": [0.05, 0.1],
}

batch_run = BatchRunnerMP(
    ModernMarchModel,
    nr_processes=CPU_COUNT,
    variable_parameters=variable_params,
    fixed_parameters=fixed_params,
    iterations=100,
    max_steps=100,
    display_progress=True,
    model_reporters={"history": track_model_steps, "ACK": calc_code_kl, "AHK": calc_human_kl}
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
df = get_tracking_data(batch_run)
print(f'Created dataframe from batch run data')
timestr = time.strftime("%Y%m%d-%H%M%S")
fname = f"{DATA_PATH}modern_march_raw_{timestr}.csv"
df.to_csv(fname)
print(f'Saved dataframe ({df.shape[0]} rows, {df.shape[1]} columns) to file {fname}')
