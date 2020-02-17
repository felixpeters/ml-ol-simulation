import datetime
from collections import Counter
from itertools import product, count

from tqdm import tqdm
import pandas as pd
from mesa.time import BaseScheduler
from mesa.batchrunner import BatchRunner

class MyBatchRunner(BatchRunner):
    def __init__(self, model_cls, **kwargs):
        super().__init__(model_cls, **kwargs)

    def get_info(self):
        num_configs = reduce(lambda prod, params: prod * len(params), self.variable_parameters.values(), 1)
        num_runs = self.iterations
        return (num_configs * num_runs, num_configs, num_runs)

    def get_tracking_data(self):
        # get number of configurations, runs and steps
        num_configs = reduce(lambda prod, params: prod * len(params), self.variable_parameters.values(), 1)
        num_runs = self.iterations
        num_steps = self.max_steps
        # create MultiIndex from cross product
        configs = list(range(1, num_configs+1))
        runs = list(range(1, num_runs+1))
        steps = list(range(0, num_steps+1))
        index = pd.MultiIndex.from_product([configs, runs, steps], names=['config', 'run', 'step'])
        # assemble data frame from model tracking data
        df = self.get_model_vars_dataframe()
        hists = df.loc[:,'history']
        res_df = pd.DataFrame()
        for hist in hists:
            hist_df = pd.DataFrame(hist.model_vars)
            hist_df = hist_df.drop(columns=["time"])
            res_df = res_df.append(hist_df)
        # reset index to created MultiIndex
        res_df.index = index
        return res_df

    def run_all(self):
        run_count = count()
        counter = 1
        start = datetime.datetime.now()
        total_iterations, all_kwargs, all_param_values = self._make_model_args()
        print('{"chart": "Progress", "axis": "Minutes"}')
        print('{"chart": "Speed", "axis": "Iterations"}')

        with tqdm(total_iterations, disable=not self.display_progress) as pbar:
            for i, kwargs in enumerate(all_kwargs):
                param_values = all_param_values[i]
                for _ in range(self.iterations):
                    self.run_iteration(kwargs, param_values, next(run_count))
                    duration = datetime.datetime.now() - start
                    seconds = duration.seconds
                    minutes = seconds / 60
                    if counter % 50 == 0:
                        print(f'{{"chart": "Progress", "y": {counter / total_iterations * 100}, "x": {minutes}}}')
                        print(f'{{"chart": "Speed", "y": {counter / seconds}, "x": {counter}}}')
                    counter += 1
                    pbar.update()
