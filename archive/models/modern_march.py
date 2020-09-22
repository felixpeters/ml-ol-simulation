import random
import datetime
from collections import Counter
from functools import partialmethod

from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

from utils.metrics import *
from utils.agents import *

class Reality(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # init randomly with -1 or 1
        self.state = random_reality(model.conf['belief_dims'])
        self.kl = 1.0

    def step(self):
        # reality is fixed
        return

class OrganizationalCode(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # init with all zeros
        self.state = np.zeros(model.conf['belief_dims'], dtype=np.int64)
        self.update_kl()

    def update_kl(self):
        self.kl = calc_kl(self.model.schedule.agents[0].state, self.state)
        return

    def learn(self):
        p_2 = self.model.conf["p_2"]
        exp_grp = self.model.get_exp_grp()
        for i in range(len(self.state)):
            # learn from expert group if existing
            exp_grp_dim = list(filter(lambda h: (h.state[i] != 0), exp_grp))
            if len(exp_grp_dim) > 0:
                votes = [h.state[i] for h in exp_grp_dim]
                c = Counter(votes)
                maj, size_maj = c.most_common()[0]
                # learn from expert group if majority belief differs from own
                if maj != self.state[i]:
                    size_min = 0
                    if len(c.most_common()) > 1: _, size_min = c.most_common()[1]
                    # update learning rate according to consensus strength
                    p_update = 1 - (1 - p_2) ** (size_maj - size_min)
                    if np.random.binomial(1, p_update): self.state[i] = maj
        return

    def step(self):
        # first learn, then update KL
        self.learn()
        self.update_kl()
        return

class Human(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # init randomly with -1, 0 or 1
        self.state = random_beliefs(model.conf['belief_dims'])
        self.update_kl()

    def update_kl(self):
        self.kl = calc_kl(self.model.schedule.agents[0].state, self.state)
        return

    def learn(self):
        code = self.model.schedule.agents[1]
        for i in range(len(self.state)):
            # if code has no knowledge: learn from data (otherwise: random)
            if code.state[i] == 0:
                self.learn_from_data(i)
            else:
                self.learn_randomly(i)
        return

    def learn_randomly(self, dim):
        # learn from data or code with equal probability
        if np.random.binomial(1, 0.5):
            self.learn_from_data(dim)
        else:
            self.learn_from_code(dim)
        return

    def learn_from_data(self, dim):
        data = self.model.schedule.agents[0]
        p_hp = self.model.conf["p_hp"]
        p_hm = self.model.conf["p_hm"]
        # 1. learn correct value with given p_h+
        if np.random.binomial(1, p_hp): 
            self.state[dim] = data.state[dim]
        else:
            # 2. learn incorrect value with given p_h-
            if np.random.binomial(1, p_hm):
                self.state[dim] = (-1) * data.state[dim]
        # 3. learn nothing otherwise
        return

    def learn_from_code(self, dim):
        code = self.model.schedule.agents[1]
        p_1 = self.model.conf["p_1"]
        # learn from code noly if code belief differs from own
        if self.state[dim] != code.state[dim] and code.state[dim] != 0:
            if np.random.binomial(1, p_1): self.state[dim] = code.state[dim]
        return

    def step(self):
        self.learn()
        self.update_kl()

class ModernMarchModel(Model):

    def __init__(
            self,
            num_humans=50, 
            belief_dims=30, 
            p_1=0.1, 
            p_2=0.9,
            p_hp=0.1,
            p_hm=0.1,
        ):
        # reset random seeds prior to each iteration
        np.random.seed()
        random.seed()
        # save configuration
        self.conf = {
                "num_humans": num_humans,
                "belief_dims": belief_dims,
                "p_1": p_1,
                "p_2": p_2,
                "p_hp": p_hp,
                "p_hm": p_hm,
        }
        self.running = True
        self.schedule = BaseScheduler(self)
        # init environment and data collector
        self.init_env()
        self.init_dc()

    def get_config(self, param, *args):
        return self.conf[param]

    # necessary in order to satisfy data collector interface
    get_belief_dims = partialmethod(get_config, "belief_dims") 
    get_num_humans = partialmethod(get_config, "num_humans") 
    get_p_1 = partialmethod(get_config, "p_1") 
    get_p_2 = partialmethod(get_config, "p_2") 
    get_p_hp = partialmethod(get_config, "p_hp") 
    get_p_hm = partialmethod(get_config, "p_hm") 

    def get_time(self, *args):
        return int(self.schedule.time)

    def init_env(self):
        # init reality
        r = Reality("R1", self) 
        self.schedule.add(r)
        # init organization
        o = OrganizationalCode("O1", self)
        self.schedule.add(o)
        # init humans
        for i in range(self.conf["num_humans"]):
            h = Human(f"H{i+1}", self)
            self.schedule.add(h)
        return

    def init_dc(self):
        # data collector enables tracking of metric at each time step
        self.datacollector = DataCollector(
                model_reporters = {
                    "time": self.get_time,
                    "belief_dims": self.get_belief_dims,
                    "num_humans": self.get_num_humans,
                    "p_1": self.get_p_1,
                    "p_2": self.get_p_2,
                    "p_hp": self.get_p_hp,
                    "p_hm": self.get_p_hm,
                    "code_kl": calc_code_kl,
                    "human_kl": calc_human_kl,
                    "human_kl_var": calc_kl_var,
                    "human_kl_dissim": calc_dissim,
                }
        )
        # collect metrics for time step 0
        self.datacollector.collect(self)
        return

    def get_exp_grp(self):
        # get list of humans with higher KL than code
        humans = self.schedule.agents[2:(2 + self.conf["num_humans"])]
        code = self.schedule.agents[1]
        return list(filter(lambda h: (h.kl > code.kl), humans))

    def step(self):
        try:
            # determine expert group for this time step
            self.exp_grp = self.get_exp_grp()
            # update all agents
            self.schedule.step()
            # collect metrics for this time step
            self.datacollector.collect(self)
        except Exception as e:
            # log potential erros, but continue with next iteration
            print("The following error occurred:")
            print(e)
            print("Model configuration:")
            print(self.conf)
