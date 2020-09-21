import time
import os

from mesa.batchrunner import BatchRunnerMP

from utils.runners import get_info, get_tracking_data
from utils.metrics import *
from utils.analysis import preprocess_dataset
from utils.params import fixed_params, test_params, full_params
from models.revision_2_model import Revision2Model

if __name__ == '__main__':

    # collected data will be saved in this folder
    DATA_PATH = "data/"
    # get the number of available CPUs for multi-processing
    CPU_COUNT = os.cpu_count() or 2
    variable_params = test_params
    
    batch_run = BatchRunnerMP(
        Revision2Model,
        nr_processes=CPU_COUNT,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=1, # =80 for production
        max_steps=100, # =200 for production
        display_progress=True,
        model_reporters={
            "history": track_model_steps, 
        },
    )
    

    # simulation batch run
    total_iter, num_conf, num_iter = get_info(batch_run, variable_params)
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
        "m": "mean",
        "n": "mean",
        "j": "mean",
        "p_1": "mean",
        "p_2": "mean",
        "p_3": "mean",
        "q_h1": "mean",
        "q_h2": "mean",
        "q_ml": "mean",
        "q_d": "mean",
        "alpha_d": "mean",
        "alpha_ml": "mean",
        "p_turb": "mean",
        "q_ml_scaling": "last",
        "q_d_scaling": "last",
        "avg_q_d": "mean",
        "avg_q_ml": "mean",
        "code_kl": ["mean", "std"],
        "human_kl": ["mean", "std"],
        "human_kl_var": "mean",
        "human_kl_dissim": "mean",
        "data_qual": "mean",
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
        "q_d": "mean",
        "alpha_d": "mean",
        "alpha_ml": "mean",
        "p_turb": "mean",
        "q_ml_scaling": "last",
        "q_d_scaling": "last",
        "avg_q_d": ["max", "last"],
        "avg_q_ml": ["max", "last"],
        "code_kl": ["max", "last"],
        "human_kl": ["max", "last"],
        "human_kl_var": ["max", "last"],
        "code_kl_std": "last",
        "human_kl_std": "last",
        "human_kl_dissim": ["max", "last"],
        "data_qual": ["max", "last"],
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
        "q_d_mean": "q_d",
        "alpha_d_mean": "alpha_d",
        "alpha_ml_mean": "alpha_ml",
        "p_turb_mean": "p_turb",
        "avg_q_d_mean": "avg_q_d",
        "avg_q_ml_mean": "avg_q_ml",
        "code_kl_mean": "code_kl",
        "q_ml_scaling_last": "q_ml_scaling",
        "q_d_scaling_last": "q_d_scaling",
        "human_kl_mean": "human_kl",
        "human_kl_var_mean": "human_kl_var",
        "human_kl_dissim_mean": "human_kl_dissim",
        "code_kl_std_last": "code_kl_std",
        "human_kl_std_last": "human_kl_std",
        "data_qual_mean": "data_qual",
    }
    time_data, agg_data = preprocess_dataset(df, run_aggs, time_aggs, col_names)
    time_fname = f"{DATA_PATH}r2_ts_{timestr}.csv"
    time_data.to_csv(time_fname)
    print(f'Saved time-series dataframe ({time_data.shape[0]} rows, {time_data.shape[1]} columns) to file {time_fname}')
    agg_fname = f"{DATA_PATH}r2_agg_{timestr}.csv"
    agg_data.to_csv(agg_fname)
    print(f'Saved aggregated dataframe ({agg_data.shape[0]} rows, {agg_data.shape[1]} columns) to file {agg_fname}')
