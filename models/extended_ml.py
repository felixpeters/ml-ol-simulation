import random
import datetime
import math
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

    def turbulence(self):
        p_turb = self.model.conf["p_turb"]
        for i in range(len(self.state)):
            if np.random.binomial(1, p_turb):
                self.state[i] = (-1) * self.state[i]
        return

    def step(self):
        self.turbulence()
        return

class OrganizationalCode(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # init with all zeros
        self.state = np.zeros(model.conf['belief_dims'], dtype=np.int64)
        self.update_kl()

    def update_kl(self):
        reality = self.model.get_reality()
        self.kl = calc_kl(reality.state, self.state)
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
        reality = self.model.get_reality()
        self.kl = calc_kl(reality.state, self.state)
        return

    def learn(self):
        for dim in range(len(self.state)):
            # learn from data or code with equal probability
            if np.random.binomial(1, 0.5):
                self.learn_from_data(dim)
            else:
                self.learn_from_code(dim)
        return

    def learn_from_data(self, dim):
        data = self.model.get_reality()
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
        code = self.model.get_org_code()
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

    def __init__(self, unique_id, model, dim):
        super().__init__(unique_id, model)
        # state consists of non-zero dimension and current value
        self.state = {
            "dim": dim,
            "val": 1 if np.random.binomial(1, 0.5) else -1,
        }
        self.update_kl()

    def update_kl(self):
        reality = self.model.get_reality()
        # KL is binary: 1 if corresponding to reality, 0 otherwise
        self.kl = 1.0 if self.state["val"] == reality.state[self.state["dim"]] else 0.0
        return

    def learn(self):
        real_val = self.model.get_reality()
        real_val = reality.state[self.state["dim"]]
        #get p_ml depending on p_ml_scaling
        if self.model.get_p_ml_scaling() == "logistic" or self.model.get_p_ml_scaling() == "march_like":
            p_ml = self.model.conf["p_ml"][self.model.conf["ml_dims"].index(self.state["dim"])]
            #prob = p_ml[self.model.conf["ml_dims"].index(self.state["dim"])]
        else:
            p_ml = self.model.conf["p_ml"]
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

class ExtendedMLModel(Model):

    def __init__(
            self,
            num_humans=50, 
            num_ml=5, 
            belief_dims=30, 
            p_1=0.1, 
            p_2=0.9,
            p_3=0.9,
            p_h1=0.1,
            p_h2=0.1,
            p_ml=0.5,
            p_turb=0.1,
            p_ml_scaling="off",
        ):
        # reset random seeds prior to each iteration
        np.random.seed()
        random.seed()
        # save configuration
        self.conf = {
                "num_humans": num_humans,
                "num_ml": num_ml,
                "belief_dims": belief_dims,
                "p_1": p_1,
                "p_2": p_2,
                "p_3": p_3,
                "p_h1": p_h1,
                "p_h2": p_h2,
                "p_ml": p_ml,
                "p_ml_basic": p_ml,
                "p_turb": p_turb,
                "p_ml_scaling": p_ml_scaling,
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
    get_p_1 = partialmethod(get_config, "p_1") 
    get_p_2 = partialmethod(get_config, "p_2") 
    get_p_3 = partialmethod(get_config, "p_3") 
    get_p_h1 = partialmethod(get_config, "p_h1") 
    get_p_h2 = partialmethod(get_config, "p_h2") 
    get_p_ml = partialmethod(get_config, "p_ml")
    get_p_ml_basic = partialmethod(get_config, "p_ml_basic")
    get_p_turb = partialmethod(get_config, "p_turb")
    get_p_ml_scaling = partialmethod(get_config, "p_ml_scaling") 

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
        # determine ML dimensions
        self.conf["ml_dims"] = random_dims(self.conf["belief_dims"], self.conf["num_ml"])
        # init one agent per dimension
        for i in self.conf["ml_dims"]:
            m = MLAgent(f"ML{i+1}", self, i)
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
                    "p_1": self.get_p_1,
                    "p_2": self.get_p_2,
                    "p_3": self.get_p_3,
                    "p_h1": self.get_p_h1,
                    "p_h2": self.get_p_h2,
                    "p_ml": self.get_p_ml_basic,
                    "p_turb": self.get_p_turb,
                    "p_ml_scaling": self.get_p_ml_scaling,
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
        humans = self.get_human_agents()
        code = self.get_org_code()
        return list(filter(lambda h: (h.kl > code.kl), humans))

    def get_ml(self, dim):
        # loop through MLs to find the one with suitable dimension
        mls = self.get_ml_agents()
        ml = None
        for m in mls:
            if m.state["dim"] == dim:
                ml = m
                break
        return ml

    def get_reality(self):
        return self.schedule.agents[0]

    def get_org_code(self):
        return self.schedule.agents[1]

    def get_human_agents(self):
        return self.schedule.agents[2:(2 + self.conf["num_humans"])]

    def get_ml_agents(self):
        return self.schedule.agents[(2+self.conf["num_humans"]):]

    def scale_p_ml(self):
        #for avg. human knowledge related manipulation
        method = self.conf["p_ml_scaling"]
        humans = self.get_human_agents()
        human_kl = np.mean([h.kl for h in humans])
        #for belief related manipulation
        exp_grp = self.get_exp_grp()
        num_ml = self.get_num_ml()
        ml_dims = self.conf["ml_dims"]
        if method == "coupled":
            self.conf["p_ml"] = human_kl
        if method == "logistic" or method == "march_like":
            p_ml = []
            for i in range(num_ml):
                #current dimension
                dim = ml_dims[i]
                #get knowledgeable group
                exp_grp_dim = list(filter(lambda h: (h.state[dim] != 0), exp_grp))
                #get basic parameters
                reality = self.get_reality()
                reality = reality.state[dim]
                p_ml_basic = self.conf["p_ml_basic"]

                if len(exp_grp_dim) > 0:
                    #if expert group exists, count correct and incorrect beliefs
                    votes = [h.state[dim] for h in exp_grp_dim]
                    c = Counter(votes)

                    if len(c) > 1:
                        #if expert group has correct and incorrect beliefs calculate difference
                        k = c.get(reality)-c.get((-1)*reality)
                    else:
                        if votes[0] == reality:
                            #all expert beliefs are correct
                            k = c.get(reality)
                        else:
                            #all expert beliefs are incorrerect
                            k = c.get((-1)*reality)
                    if method == "logistic":
                        #see Google Drive/Forschung/MISQ/ExtensionDesign for formulas
                        alpha = 1
                        beta = math.log((1-p_ml_basic)/p_ml_basic)
                        p_ml.append(round(1/(1+math.e**(((-1)*k/alpha)+beta)),3))
                    if method == "march_like":
                        p_ml.append(1-(1-p_ml_basic)**((k/len(humans))+1))
                else:
                    #if there are no experts, use basic p_ml value
                    p_ml.append(p_ml_basic)
            self.conf["p_ml"] = p_ml
        return

    def step(self):
        try:
            # determine expert group for this time step
            self.exp_grp = self.get_exp_grp()
            # scale p_ml according to human KL
            self.scale_p_ml()
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
