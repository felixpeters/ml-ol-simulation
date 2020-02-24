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
    "lr_hum_org": [0.1, 0.5, 0.9],
    "lr_org_hum": [0.1, 0.5, 0.9],
    "lr_hum_dat": [0.05, 0.1],
}

batch_run = BatchRunnerMP(
    ModernMarchModel,
    nr_processes=CPU_COUNT,
    variable_parameters=variable_params,
    fixed_parameters=fixed_params,
    iterations=50,
    max_steps=100,
    display_progress=False,
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
print(f'Simulation completed after {end-start:.2f} seconds')
df = get_tracking_data(batch_run)
print(f'Created dataframe from batch run data')
timestr = time.strftime("%Y%m%d-%H%M%S")
fname = f"{DATA_PATH}modern_march_{timestr}.csv"
df.to_csv(fname)
print(f'Saved dataframe ({df.shape[0]} rows, {df.shape[1]} columns) to file {fname}')
