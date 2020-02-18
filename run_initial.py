import time
from utils.runners import get_info, get_tracking_data
from utils.metrics import track_model_steps, calc_code_knowledge, calc_human_knowledge
from models.initial import InitialModel
from mesa.batchrunner import BatchRunner

# constants
DATA_PATH = "data/"
#DATA_PATH="/storage/"

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

batch_run = BatchRunner(
    InitialModel,
    variable_parameters=variable_params,
    fixed_parameters=fixed_params,
    iterations=5,
    max_steps=50,
    display_progress=False,
    model_reporters={"history": track_model_steps, "ACK": calc_code_knowledge, "AHK": calc_human_knowledge}
)

# simulation batch run
total_iter, num_conf, num_iter = get_info(batch_run)
print(f'Starting simulation with a total of {total_iter} iterations ({num_conf} configurations, {num_iter} iterations per configuration)...')
start = time.time()
batch_run.run_all()
end = time.time()
print(f'Simulation complete after {end-start:.2f} seconds')
print(f'Creating data frame from batch run data...')
df = get_tracking_data(batch_run)
print(f'Saving data frame ({df.shape[0]} rows, {df.shape[1]} columns) to file...')
timestr = time.strftime("%Y%m%d-%H%M%S")
df.to_csv(f"{DATA_PATH}ai_simulation_{timestr}.csv")
