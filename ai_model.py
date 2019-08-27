import random
import datetime
from collections import Counter
from functools import partial, partialmethod, reduce
from itertools import product, count

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

# constants
DATA_PATH="/storage/"

# helper functions
def random_beliefs(k):
    a = np.random.randint(-1, 2, k)
    return a

def random_reality(k):
    a = np.random.randint(0, 2, k)
    a[a == 0] = -1
    return a

def calc_knowledge_level(reality, beliefs):
    equals = 0
    for i in range(len(reality)):
        if reality[i] == beliefs[i]: equals += 1
    res = float(equals) / float(len(reality))
    return res

def calc_code_knowledge(model):
    return model.schedule.agents[1].kl

def calc_avg_knowledge(model):
    humans = model.human_agents(active_only=True)
    kls = [h.kl for h in humans]
    return np.mean(kls)

def track_model_steps(model):
    return model.datacollector

def calc_ai_knowledge(model):
    agents = model.ai_agents()
    total_ai_dims = len(model.ai_dimensions)
    correct_ai_dims = 0
    for agent in agents:
        belief = agent.beliefs[agent.belief_dimension]
        truth = model.schedule.agents[0].state[agent.belief_dimension]
        if belief == truth: correct_ai_dims += 1
    if total_ai_dims == 0:
        return 0.0
    else:
        return correct_ai_dims / total_ai_dims

def random_transparency():
    return np.random.random()

def fixed_transparency():
    return 0.5

def high_transparency():
    return 0.8

def low_transparency():
    return 0.2

def full_transparency():
    return 1.0

def get_tracking_data_from_batch(batch_runner):
    # get number of configurations, runs and steps
    num_configs = reduce(lambda prod, params: prod * len(params), batch_run.variable_parameters.values(), 1)
    num_runs = batch_run.iterations
    num_steps = batch_run.max_steps
    # create MultiIndex from cross product
    configs = list(range(1, num_configs+1))
    runs = list(range(1, num_runs+1))
    steps = list(range(0, num_steps+1))
    index = pd.MultiIndex.from_product([configs, runs, steps], names=['config', 'run', 'step'])
    # assemble data frame from model tracking data
    df = batch_run.get_model_vars_dataframe()
    hists = df.loc[:,'history']
    res_df = pd.DataFrame()
    for hist in hists:
        hist_df = pd.DataFrame(hist.model_vars)
        hist_df = hist_df.drop(columns=["time"])
        res_df = res_df.append(hist_df)
    # reset index to created MultiIndex
    res_df.index = index
    return res_df

def get_batch_run_info(batch_runner):
    num_configs = reduce(lambda prod, params: prod * len(params), batch_run.variable_parameters.values(), 1)
    num_runs = batch_run.iterations
    return (num_configs * num_runs, num_configs, num_runs)

# model classes
class Human(Agent):
    """ A human agent with k-dimensional beliefs. """
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # initialize belief vector (defaults to 30)
        self.beliefs = random_beliefs(model.conf['belief_dimensions'])
        self.active = True
        self.belief_history = self.beliefs
        self.p1 = model.conf["p1"]
        self.update_kl()
        
    def __repr__(self):
        return "Human(id={}, active={}, beliefs={}, kl={:.2f}, p1={:.2f})".format(
            self.unique_id, 
            self.active, 
            self.beliefs, 
            self.kl,
            self.p1,
        )
    
    def update_kl(self):
        self.kl = calc_knowledge_level(self.model.schedule.agents[0].state, self.beliefs)
        
    def update_belief_history(self):
        if self.active:
            self.belief_history = np.vstack((self.belief_history, self.beliefs))
        else:
            self.belief_history = np.vstack((self.belief_history, np.zeros(len(self.beliefs), dtype=int)))
            
    def get_belief_history(self, window=None):
        if (window != None) and (len(self.belief_history.shape) > 1):
            return self.belief_history[-window:]
        return self.belief_history
        
    def deactivate(self):
        self.active = False
        
    def increase_exploration(self):
        exp_inc = self.model.conf["exploration_increase"]
        self.p1 = self.p1 * (1 - exp_inc)
        
    def learn_from_code(self):
        code = self.model.schedule.agents[1].code
        p1 = self.p1
        # loop over code and belief dimensions
        for i in range(len(code)):
            # if code differs from belief AND code is not zero, then try to update belief
            if code[i] != self.beliefs[i] and code[i] != 0:
                # if this is an AI dimension, multiply p1 with AI's transparency level
                if i in self.model.ai_dimensions:
                    agent = self.model.ai_agent(i)
                    p1 = p1 * agent.transparency
                # update belief with probability p1
                if np.random.binomial(1, p1):
                    self.beliefs[i] = code[i]
                    
    def turnover(self):
        p3 = self.model.conf["p3"]
        if p3 > 0.0 and np.random.binomial(1, p3):
            self.beliefs = random_beliefs(self.model.conf['belief_dimensions'])
    
    def step(self):
        if self.active:
            self.turnover()
            self.learn_from_code()
            self.update_kl()
        self.update_belief_history()

class Organization(Agent):
    """ An organization with k-dimensional code. """
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # initialize belief vector (defaults to 30)
        self.code = np.zeros(model.conf['belief_dimensions'], dtype=np.int64)
        self.update_kl()
        
    def __repr__(self):
        return "Organization(id={}, code={}, kl={:.2f})".format(self.unique_id, self.code, self.kl)
        
    def update_kl(self):
        self.kl = calc_knowledge_level(self.model.schedule.agents[0].state, self.code)
        
    def learn(self):
        for i in range(len(self.code)):
            # if dimension is covered by AI, learn from AI
            if i in self.model.ai_dimensions:
                self.learn_from_ai(i)
            else:
                self.learn_from_humans(i)
    
    def learn_from_ai(self, dimension):
        ai = self.model.ai_agent(dimension)
        if ai.beliefs[dimension] != self.code[dimension]:
            self.code[dimension] = ai.beliefs[dimension]
        
    def learn_from_humans(self, dimension):
        p2 = self.model.conf["p2"]
        # filter humans with higher knowledge level than code that also have knowledge about dimension
        dim_humans = list(filter(lambda h: (h.kl > self.kl) and (h.beliefs[dimension] != 0), 
                                self.model.human_agents(active_only=True)))
        if len(dim_humans) > 0:
            # determine human majority for dimension
            votes = [h.beliefs[dimension] for h in dim_humans]
            c = Counter(votes)
            # get majority value and size of majority group
            maj, num_maj = c.most_common()[0]
            # only continue if majority value differs from current code value
            if maj != self.code[dimension]:
                # get size of minority group
                num_min = 0
                if len(c.most_common()) > 1: _, num_min = c.most_common()[1]
                # calculate probability that code changes
                p_update = 1 - (1 - p2) ** (num_maj - num_min)
                # update code with calculated probability
                if np.random.binomial(1, p_update): self.code[dimension] = maj
        
    def step(self):
        self.learn()
        self.update_kl()

class Reality(Agent):
    """ An organization with k-dimensional code. """
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # initialize belief vector (defaults to 30)
        self.state = random_reality(model.conf['belief_dimensions'])
        self.kl = 1.0
        
    def __repr__(self):
        return "Reality(id={}, code={})".format(self.unique_id, self.state)
    
    def environmental_turbulence(self):
        p4 = self.model.conf["p4"]
        if p4 > 0.0:
            for dim in range(len(self.state)):
                if np.random.binomial(1, p4):
                    self.state[dim] = self.state[dim] * (-1)
        
    def step(self):
        self.environmental_turbulence()

class ArtificialIntelligence(Agent):
    """ An artificial agent specialized on one belief dimension."""
    
    def __init__(self, unique_id, model, belief_dimension, transparencyFn):
        super().__init__(unique_id, model)
        self.beliefs = np.zeros(model.conf['belief_dimensions'], dtype=np.int64)
        self.belief_dimension = belief_dimension
        self.transparency = transparencyFn()
        self.lifetime = 0
        self.learn_from_humans()
        self.update_kl()
        
    def update_kl(self):
        self.kl = calc_knowledge_level(self.model.schedule.agents[0].state, self.beliefs)
        
    def learn_from_humans(self):
        window = self.model.conf["retrain_window"]
        belief_history = self.model.belief_history(self.belief_dimension, window=window)
        # determine majority value
        c = Counter(belief_history)
        maj, _ = c.most_common()[0]
        # update belief dimension with majority value
        self.beliefs[self.belief_dimension] = maj
        
    def __repr__(self):
        return "AI(id={}, belief_dim={}, beliefs={}, transparency={:.2f})".format(
            self.unique_id, 
            self.belief_dimension, 
            self.beliefs,
            self.transparency)
    
    def step(self):
        self.lifetime += 1
        freq = self.model.conf["retrain_freq"]
        window = self.model.conf["retrain_window"]
        if (freq != None) and (self.lifetime % freq == 0):
            self.learn_from_humans()


class ExtendedAIModel(Model):
    """ Extended AI model with feature toggling. """
    
    # initialization functions
    def __init__(self, 
                 num_agents=50, 
                 belief_dimensions=30, 
                 p1=0.5, 
                 p2=0.5,
                 p3=0.1,
                 p4=0.1,
                 pai=0.05,
                 p_replace=0.5,
                 transparency_fn=random_transparency,
                 retrain_freq=2,
                 retrain_window=10,
                 ai_init_mode="significant", 
                 replacement_mode="least_knowledgeable", 
                 exploration_increase=0.0):
        np.random.seed()
        self.conf = {
            "num_agents": num_agents,
            "belief_dimensions": belief_dimensions,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "pai": pai,
            "p_replace": p_replace,
            "transparency_func": transparency_fn,
            "retrain_freq": retrain_freq,
            "retrain_window": retrain_window,
            "ai_init_mode": ai_init_mode,
            "replacement_mode": replacement_mode,
            "exploration_increase": exploration_increase,
        }
        self.running = True
        self.schedule = BaseScheduler(self)
        self.init_environment()
        self.init_datacollector()
        
    def init_environment(self):
        # initialize reality
        r = Reality("R{}".format(1), self)
        self.schedule.add(r)
        # initialize organization
        o = Organization("O{}".format(1), self)
        self.schedule.add(o)
        # initialize agents
        self.human_dimensions = list(range(0, self.conf["belief_dimensions"]))
        self.ai_dimensions = []
        for i in range(self.conf['num_agents']):
            a = Human("H{}".format(i+1), self)
            self.schedule.add(a)
    
    def init_datacollector(self):
        self.datacollector = DataCollector(
            model_reporters={
                "ACK": calc_code_knowledge, 
                "AHK": calc_avg_knowledge,
                "AAIK": calc_ai_knowledge,
                "p1": self.p1,
                "p2": self.p2,
                "p3": self.p3,
                "p4": self.p4,
                "pai": self.pai,
                "p_replace": self.p_replace,
                "belief_dims": self.belief_dims,
                "human_agents": self.num_active_humans,
                "ai_agents": self.num_ais,
                "time": self.current_time,
                "transparency_fn": self.transparency_fn,
                "retrain_freq": self.retrain_freq,
                "retrain_window": self.retrain_window,
                "ai_init_mode": self.ai_init_mode,
                "replacement_mode": self.replacement_mode,
                "exploration_increase": self.exp_inc,
                "avg_p1": self.avg_p1,
            },
            agent_reporters={"knowledge_level": "kl"})
        self.datacollector.collect(self)

    # getter functions
    def config(self, param, *args):
        return self.conf[param]
    
    def transparency_fn(self, *args):
        return self.conf["transparency_func"].__name__    
    
    def current_time(self, *args):
        return int(self.schedule.time)
        
    def human_agents(self, active_only=False, num_only=False, *args):
        humans = self.schedule.agents[2:(2 + self.conf["num_agents"])]
        if active_only:
            humans= list(filter(lambda h: h.active == True, humans))
        if num_only:
            return len(humans)
        return humans
    
    def ai_agents(self, num_only=False, *args):
        agents = []
        if len(self.schedule.agents) > (2 + self.conf["num_agents"]):
            agents = self.schedule.agents[(2 + self.conf["num_agents"]):]
        if num_only:
            return len(agents)
        return agents
    
    def ai_agent(self, belief_dim):
        ai_agents = self.ai_agents()
        for agent in ai_agents:
            if agent.belief_dimension == belief_dim:
                return agent
        return None
    
    def belief_history(self, dim, window=None):
            belief_history = np.vstack([h.get_belief_history(window=window) for h in self.human_agents()])
            dim_hist = belief_history[:, dim]
            dim_hist = dim_hist[dim_hist != 0]
            return dim_hist
    
    def avg_p1(self, *args):
        p1_vals = [h.p1 for h in self.human_agents(active_only=True)]
        return np.mean(p1_vals)
    
    # partials for tracking
    p1 = partialmethod(config, "p1")
    p2 = partialmethod(config, "p2")
    p3 = partialmethod(config, "p3")
    p4 = partialmethod(config, "p4")
    pai = partialmethod(config, "pai")
    p_replace = partialmethod(config, "p_replace")
    ai_init_mode = partialmethod(config, "ai_init_mode")
    retrain_freq = partialmethod(config, "retrain_freq")
    retrain_window = partialmethod(config, "retrain_window")
    replacement_mode = partialmethod(config, "replacement_mode")
    exp_inc = partialmethod(config, "exploration_increase")
    belief_dims = partialmethod(config, "belief_dimensions")
    active_human_agents = partialmethod(human_agents, True)
    num_active_humans = partialmethod(human_agents, True, True)
    num_ais = partialmethod(ai_agents, True)
    
    # AI initialization
    def init_ais(self):
        total_dims = self.conf["belief_dimensions"]
        if len(self.ai_dimensions) < total_dims:
            init_mode = self.conf["ai_init_mode"]
            if init_mode == "random":
                self.init_random_ai()
            elif init_mode == "significant":
                self.init_significant_ai()
            
    def init_significant_ai(self):
        for dim in self.human_dimensions:
            # determine whether significant majority belief exists
            hist = self.belief_history(dim, window=self.conf["retrain_window"])
            rand = random_reality(len(hist))
            if stats.ttest_ind(hist, rand).pvalue <= 0.05:
                self.add_ai(dim)
                self.replace_human()
            
    def init_random_ai(self):
        humans = self.human_agents(active_only=True)
        for h in humans:
            if np.random.binomial(1, self.conf["pai"]):
                if len(self.human_dimensions) > 0:
                    new_dim = random.choice(self.human_dimensions)
                    self.add_ai(new_dim)
                    h.deactivate()
    
    def add_ai(self, dim):
        ai = ArtificialIntelligence("AI{}".format(len(self.ai_dimensions)+1), 
                                    self, 
                                    dim, 
                                    self.conf["transparency_func"])
        self.schedule.add(ai)
        self.ai_dimensions.append(dim)
        self.human_dimensions.remove(dim)
        
    def replace_human(self):
        replace_mode = self.conf["replacement_mode"]
        if replace_mode == "random":
            self.replace_random_human()
        elif replace_mode == "least_knowledgeable":
            self.replace_least_knowledgeable()
        elif replace_mode == "increase_exploration":
            self.increase_exploration()
            
    def replace_random_human(self):
        if np.random.binomial(1, self.conf["p_replace"]):
            humans = self.human_agents(active_only=True)
            h = random.choice(humans)
            h.deactivate()
        
    def replace_least_knowledgeable(self):
        if np.random.binomial(1, self.conf["p_replace"]):
            humans = self.human_agents(active_only=True)
            humans.sort(key=lambda h: h.kl)
            h = humans[0]
            h.deactivate()
        
    def increase_exploration(self):
        humans = self.human_agents(active_only=True)
        h = random.choice(humans)
        h.increase_exploration()
            
    def step(self):
        try:
            self.init_ais()
            self.schedule.step()
            self.datacollector.collect(self)
        except Exception as e:
            print("The following error occurred:")
            print(e)
            print("Model configuration:")
            print(self.conf)

# Batch runner
class MyBatchRunner(BatchRunner):
    def __init__(self, model_cls, **kwargs):
        super().__init__(model_cls, **kwargs)

    def run_all(self):
        run_count = count()
        counter = 1
        start = datetime.datetime.now()
        total_iterations, all_kwargs, all_param_values = self._make_model_args()
        print('{"chart": "Progress", "axis": "Minutes"}')

        with tqdm(total_iterations, disable=not self.display_progress) as pbar:
            for i, kwargs in enumerate(all_kwargs):
                param_values = all_param_values[i]
                for _ in range(self.iterations):
                    self.run_iteration(kwargs, param_values, next(run_count))
                    duration = datetime.datetime.now() - start
                    duration = duration.seconds / 60
                    print(f'{{"chart": "Progress", "y": {counter / total_iterations * 100}, "x": {duration}}}')
                    counter += 1
                    pbar.update()

# batch run configuration
fixed_params = {
    "belief_dimensions": 30,
    "num_agents": 50,
    "ai_init_mode": "significant",
    "pai": 0.05,
}

variable_params = {
    "p1": [0.2, 0.5, 0.8],
    "p2": [0.2, 0.5, 0.8],
    "p3": [0.0, 0.1],
    "p4": [0.0, 0.1],
    "p_replace": [0.5, 1.0],
    "retrain_freq": [None, 2],
    "retrain_window": [None, 5],
    "replacement_mode": ["least_knowledgeable", "increase_exploration"],
    "exploration_increase": [0.1, 0.2],
    "transparency_fn": [random_transparency, low_transparency, high_transparency, full_transparency],
}

batch_run = MyBatchRunner(
    ExtendedAIModel,
    variable_parameters=variable_params,
    fixed_parameters=fixed_params,
    iterations=1,
    max_steps=20,
    display_progress=False,
    model_reporters={"history": track_model_steps, "ACK": calc_code_knowledge, "AHK": calc_avg_knowledge}
)

# simulation batch run
total_iter, num_conf, num_iter = get_batch_run_info(batch_run)
print(f'Starting simulation with a total of {total_iter} iterations ({num_conf} configurations, {num_iter} iterations per configuration)...')
batch_run.run_all()
print(f'Creating data frame from batch run data...')
df = get_tracking_data_from_batch(batch_run)
print(f'Saving data frame ({df.shape[0]} rows, {df.shape[1]} columns) to file...')
df.to_csv(f"{DATA_PATH}simulation_data.csv")

