import time
import os

from mesa.batchrunner import BatchRunnerMP

from utils.runners import get_info, get_tracking_data, track_model_steps
from utils.analysis import preprocess_dataset, run_aggs, time_aggs, col_names
from utils.params import test_config, run_config
from models.base_model import BaseModel

if __name__ == '__main__':

    # define constants
    DATA_PATH = "data/"
    MODEL_NAME = "base"
    CPU_COUNT = os.cpu_count() or 2
    CONFIG = test_config

    # create multi-process runner
    batch_run = BatchRunnerMP(
        BaseModel,
        nr_processes=CPU_COUNT,
        variable_parameters=CONFIG["variable_params"],
        fixed_parameters=CONFIG["fixed_params"],
        iterations=CONFIG["num_iterations"],
        max_steps=CONFIG["num_steps"],
        display_progress=True,
        model_reporters={
            "history": track_model_steps,
        },
    )

    # run simulation
    total_iter, num_conf, num_iter = get_info(
        batch_run, CONFIG["variable_params"])
    print(f'Starting simulation with the following setup:')
    print(f'- Total number of iterations: {total_iter}')
    print(f'- Number of configurations: {num_conf}')
    print(f'- Iterations per configuration: {num_iter}')
    print(f'- Number of processing cores: {CPU_COUNT}')
    start = time.time()
    batch_run.run_all()
    end = time.time()
    duration = end - start
    print(
        f'Simulation completed after {duration:.2f} seconds (speed: {total_iter/duration:.2f} iterations/second)')

    # get tracking data
    df = get_tracking_data(batch_run)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(
        f'Created raw dataframe from batch run data with following shape: {df.shape[0]} rows, {df.shape[1]} columns')

    # preprocess data
    time_data, agg_data = preprocess_dataset(
        df, run_aggs, time_aggs, col_names)
    time_fname = f"{DATA_PATH}{MODEL_NAME}_ts_{timestr}.csv"
    time_data.to_csv(time_fname)
    print(
        f'Saved time-series dataframe ({time_data.shape[0]} rows, {time_data.shape[1]} columns) to file {time_fname}')
    agg_fname = f"{DATA_PATH}{MODEL_NAME}_agg_{timestr}.csv"
    agg_data.to_csv(agg_fname)
    print(
        f'Saved aggregated dataframe ({agg_data.shape[0]} rows, {agg_data.shape[1]} columns) to file {agg_fname}')
