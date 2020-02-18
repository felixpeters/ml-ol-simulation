import time
import os

from mesa.batchrunner import BatchRunnerMP

from utils.runners import get_info, get_tracking_data
from utils.metrics import track_model_steps, calc_code_knowledge, calc_human_knowledge
from models.initial import InitialModel

# constants
DATA_PATH = "data/"
CPU_COUNT = os.cpu_count() or 2

# batch run configuration
fixed_params = {
    "belief_dimensions": 30,
    "num_agents": 50,
    "retrain_freq": 1,
    "retrain_window": None,
    "learning_strategy": "balanced",
    "turbulence": "on",
    "exploration_increase": "off",
    "required_majority": 0.8,
}

variable_params = {
    "transparency": [0.5, 0.9],
}

batch_run = BatchRunnerMP(
    InitialModel,
    nr_processes=CPU_COUNT,
    variable_parameters=variable_params,
    fixed_parameters=fixed_params,
    iterations=5,
    max_steps=50,
    display_progress=False,
    model_reporters={"history": track_model_steps, "ACK": calc_code_knowledge, "AHK": calc_human_knowledge}
)

# simulation batch run
total_iter, num_conf, num_iter = get_info(batch_run)
print(f'Starting simulation with the following setup:\n- Total number of
        iterations: {total_iter}\n- Number of configurations: {num_conf}\n-
        Iterations per configuration: {num_iter}\n- Number of processing cores: {CPU_COUNT}')
start = time.time()
batch_run.run_all()
end = time.time()
print(f'Simulation completed after {end-start:.2f} seconds')
df = get_tracking_data(batch_run)
print(f'Created dataframe from batch run data')
timestr = time.strftime("%Y%m%d-%H%M%S")
fname = f"{DATA_PATH}ai_simulation_{timestr}.csv"
df.to_csv(fname)
print(f'Saved dataframe ({df.shape[0]} rows, {df.shape[1]} columns) to file {fname}')
