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
        exp_grp = self.model.get_exp_grp()
        ml_dims = self.model.conf["ml_dims"]
        for i in range(len(self.state)):
            exp_grp_dim = list(filter(lambda h: (h.state[i] != 0), exp_grp))
            # if ML is present and expert group has beliefs: learn randomly
            if (i in ml_dims) and (len(exp_grp_dim) > 0):
                if np.random.binomial(1, 0.5):
                    self.learn_from_ml(i)
                else:
                    self.learn_from_humans(i, exp_grp_dim)
            # if expert group has no belief: always learn from ML
            elif (i in ml_dims) and (len(exp_grp_dim) == 0):
                self.learn_from_ml(i)
            # if no ML is present: always learn from humans
            elif (i not in ml_dims) and (len(exp_grp_dim) > 0):
                self.learn_from_humans(i, exp_grp_dim)
            # if no ML is present and expert group has no belief: do nothing
        return

    def learn_from_humans(self, dim, exp_grp_dim):
        p_2 = self.model.conf["p_2"]
        if len(exp_grp_dim) > 0:
            votes = [h.state[dim] for h in exp_grp_dim]
            c = Counter(votes)
            maj, size_maj = c.most_common()[0]
            # learn from expert group if majority belief differs from own
            if maj != self.state[dim]:
                size_min = 0
                if len(c.most_common()) > 1: _, size_min = c.most_common()[1]
                # update learning rate according to consensus strength
                p_update = 1 - (1 - p_2) ** (size_maj - size_min)
                if np.random.binomial(1, p_update): self.state[dim] = maj
        return

    def learn_from_ml(self, dim):
        p_3 = self.model.conf["p_3"]
        ml = self.model.get_ml(dim)
        # learn from ML if existing
        if ml is not None:
            ml_val = ml.state["val"]
            # learn from ML if its belief differs from own
            if ml_val != self.state[dim]:
                if np.random.binomial(1, p_3): self.state[dim] = ml_val
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
        for dim in range(len(self.state)):
            # learn from data or code with equal probability
            if np.random.binomial(1, 0.5):
                self.learn_from_data(dim)
            else:
                self.learn_from_code(dim)
        return

    def learn_from_data(self, dim):
        data = self.model.schedule.agents[0]
        p_h1 = self.model.conf["p_h1"]
        p_h2 = self.model.conf["p_h2"]
        # learn with probability p_h1
        if np.random.binomial(1, p_h1): 
            # learn correct value with probability p_h2, incorrect value
            # otherwise
            if np.random.binomial(1, p_h2):
                self.state[dim] = data.state[dim]
            else:
                self.state[dim] = (-1) * data.state[dim]
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
        return

class MLAgent(Agent):

    def __init__(self, unique_id, model, dim, p_ml):
        super().__init__(unique_id, model)
        # state consists of non-zero dimension and current value
        self.state = {
            "dim": dim,
            "val": 1 if np.random.binomial(1, 0.5) else -1,
        }
        self.p_ml = p_ml
        self.update_kl()

    def update_kl(self):
        reality = self.model.schedule.agents[0].state
        # KL is binary: 1 if corresponding to reality, 0 otherwise
        self.kl = 1.0 if self.state["val"] == reality[self.state["dim"]] else 0.0
        return

    def learn(self):
        p_ml = self.p_ml
        real_val = self.model.schedule.agents[0].state[self.state["dim"]]
        # adopt reality value with p_ml, otherwise adopt incorrect value
        if np.random.binomial(1, p_ml):
            self.state["val"] = real_val
        else:
            self.state["val"] = (-1) * real_val
        return

    def step(self):
        self.learn()
        self.update_kl()
        return

class BasicMLModel(Model):

    def __init__(
            self,
            num_humans=50, 
            num_ml=5, 
            num_bad_ml=5, 
            belief_dims=30, 
            p_1=0.1, 
            p_2=0.9,
            p_3=0.9,
            p_h1=0.1,
            p_h2=0.1,
            p_ml=0.5,
            p_ml_bad=0.2,
        ):
        # reset random seeds prior to each iteration
        np.random.seed()
        random.seed()
        # save configuration
        self.conf = {
                "num_humans": num_humans,
                "num_ml": num_ml,
                "num_bad_ml": num_bad_ml,
                "belief_dims": belief_dims,
                "p_1": p_1,
                "p_2": p_2,
                "p_3": p_3,
                "p_h1": p_h1,
                "p_h2": p_h2,
                "p_ml": p_ml,
                "p_ml_bad": p_ml_bad,
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
    get_num_ml = partialmethod(get_config, "num_ml") 
    get_num_bad_ml = partialmethod(get_config, "num_bad_ml") 
    get_p_1 = partialmethod(get_config, "p_1") 
    get_p_2 = partialmethod(get_config, "p_2") 
    get_p_3 = partialmethod(get_config, "p_3") 
    get_p_h1 = partialmethod(get_config, "p_h1") 
    get_p_h2 = partialmethod(get_config, "p_h2") 
    get_p_ml = partialmethod(get_config, "p_ml") 
    get_p_ml_bad = partialmethod(get_config, "p_ml_bad") 

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
        # init ML agents
        # determine ML dimensions and number of good/bad MLs
        self.conf["ml_dims"] = random_dims(self.conf["belief_dims"], self.conf["num_ml"])
        num_good_ml = self.conf["num_ml"] - self.conf["num_bad_ml"]
        num_bad_ml = self.conf["num_bad_ml"]
        # init good MLs
        for i in range(num_good_ml):
            dim = self.conf["ml_dims"][i]
            m = MLAgent(f"ML{dim}", self, dim, self.conf["p_ml"])
            self.schedule.add(m)
        # init bad MLs
        for i in range(num_good_ml, len(self.conf["ml_dims"]))
            dim = self.conf["ml_dims"][i]
            m = MLAgent(f"ML{dim}", self, dim, self.conf["p_ml_bad"])
            self.schedule.add(m)
        return

    def init_dc(self):
        # data collector enables tracking of metric at each time step
        self.datacollector = DataCollector(
                model_reporters = {
                    "time": self.get_time,
                    "belief_dims": self.get_belief_dims,
                    "num_humans": self.get_num_humans,
                    "num_ml": self.get_num_ml,
                    "num_bad_ml": self.get_num_bad_ml,
                    "p_1": self.get_p_1,
                    "p_2": self.get_p_2,
                    "p_3": self.get_p_3,
                    "p_h1": self.get_p_h1,
                    "p_h2": self.get_p_h2,
                    "p_ml": self.get_p_ml,
                    "p_ml_bad": self.get_p_ml_bad,
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

    def get_ml(self, dim):
        # loop through MLs to find the one with suitable dimension
        mls = self.schedule.agents[(2+self.conf["num_humans"]):]
        ml = None
        for m in mls:
            if m.state["dim"] == dim:
                ml = m
                break
        return ml

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
