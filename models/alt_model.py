import random
import datetime
import math
from collections import Counter
from functools import partialmethod

import numpy as np
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

from .utils.metrics import calc_kl, calc_code_kl, calc_human_kl, calc_avg_q_ml, calc_kl_var, calc_dissim
from .utils.agents import random_beliefs, random_dims, random_reality


class Reality(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # init randomly with -1 or 1
        self.state = random_reality(model.conf['m'])
        self.kl = 1.0

    def turbulence(self):
        p_turb = self.model.conf["p_turb"]
        probs = np.random.binomial(1, p_turb, len(self.state))
        self.state = [(-1)*s if p == 1 else s for (s, p)
                      in zip(self.state, probs)]
        return

    def step(self):
        return


class OrganizationalCode(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # init with all zeros
        self.state = np.zeros(model.conf['m'], dtype=np.int64)
        self.update_kl()

    def update_kl(self):
        reality = self.model.get_reality()
        self.kl = calc_kl(reality.state, self.state)
        return

    # TODO: speed up learning loop
    def learn(self):
        exp_grp = self.model.get_exp_grp()
        for i in range(len(self.state)):
            exp_grp_dim = list(filter(lambda h: (h.state[i] != 0), exp_grp))
            # if ML is present and expert group has beliefs: learn randomly
            if len(exp_grp_dim) > 0:
                self.learn_from_humans(i, exp_grp_dim)
        return

    # TODO: speed up learning loop
    def learn_from_humans(self, dim, exp_grp_dim):
        p_2 = self.model.conf["p_2"]
        if len(exp_grp_dim) > 0:
            votes = [h.state[dim] for h in exp_grp_dim]
            c = Counter(votes)
            maj, size_maj = c.most_common()[0]
            # learn from expert group if majority belief differs from own
            if maj != self.state[dim]:
                size_min = 0
                if len(c.most_common()) > 1:
                    _, size_min = c.most_common()[1]
                # update learning rate according to consensus strength
                p_update = 1 - (1 - p_2) ** (size_maj - size_min)
                if np.random.binomial(1, p_update):
                    self.state[dim] = maj
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
        self.state = random_beliefs(model.conf['m'])
        self.update_kl()

    def update_kl(self):
        reality = self.model.get_reality()
        self.kl = calc_kl(reality.state, self.state)
        return

    # TODO: speed up learning loop
    def learn(self):
        ml_dims = self.model.conf["ml_dims"]
        for dim in range(len(self.state)):
            # if this is ML dim, learn with equal prob from data, code or ML
            if dim in ml_dims:
                choice = np.random.randint(0, 3)
                if choice == 0:
                    self.learn_from_ml(dim)
                elif choice == 1:
                    self.learn_from_data(dim)
                elif choice == 2:
                    self.learn_from_code(dim)
            # if this is no ML dim, learn with equal prob from data or code
            else:
                if np.random.binomial(1, 0.5):
                    self.learn_from_data(dim)
                else:
                    self.learn_from_code(dim)
        return

    def learn_from_data(self, dim):
        data = self.model.get_reality()
        q_h1 = self.model.conf["q_h1"]
        q_h2 = self.model.conf["q_h2"]
        # learn something with probability q_h1
        if np.random.binomial(1, q_h1):
            # learn correct value with prob q_h2, incorrect value otherwise
            if np.random.binomial(1, q_h2):
                self.state[dim] = data.state[dim]
            else:
                self.state[dim] = (-1) * data.state[dim]
        return

    def learn_from_ml(self, dim):
        p_3 = self.model.conf["p_3"]
        ml = self.model.get_ml(dim)
        # learn from ML if existing
        if ml is not None:
            ml_val = ml.state["val"]
            # learn from ML if its belief differs from own
            if ml_val != self.state[dim]:
                if np.random.binomial(1, p_3):
                    self.state[dim] = ml_val
        return

    def learn_from_code(self, dim):
        code = self.model.get_org_code()
        p_1 = self.model.conf["p_1"]
        # learn from code noly if code belief differs from own
        if self.state[dim] != code.state[dim] and code.state[dim] != 0:
            if np.random.binomial(1, p_1):
                self.state[dim] = code.state[dim]
        return

    def step(self):
        self.learn()
        self.update_kl()
        return


class MLAgent(Agent):

    # TODO: unify ML agents into single agent in order to allow speed-up
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
        data = self.model.get_reality()
        data_val = data.state[self.state["dim"]]
        # get q_ml depending on q_ml_scaling
        if self.model.get_q_ml_scaling() == "on":
            q_ml = self.model.conf["q_ml"][self.model.conf["ml_dims"].index(
                self.state["dim"])]
        else:
            q_ml = self.model.conf["q_ml"]
        # adopt reality value with q_ml, otherwise adopt incorrect value
        if np.random.binomial(1, q_ml):
            self.state["val"] = data_val
        else:
            self.state["val"] = (-1) * data_val
        return

    def step(self):
        self.learn()
        self.update_kl()
        return


class AlternativeModel(Model):

    def __init__(
        self,
        n=50,
        j=5,
        m=30,
        p_1=0.1,
        p_2=0.9,
        p_3=0.9,
        q_h1=0.1,
        q_h2=0.1,
        q_ml=0.5,
        alpha=10,
        p_turb=0.1,
        q_ml_scaling="off",
    ):
        # reset random seeds prior to each iteration
        np.random.seed()
        random.seed()
        # save configuration
        self.conf = {
            "n": n,
            "j": j,
            "m": m,
            "p_1": p_1,
            "p_2": p_2,
            "p_3": p_3,
            "q_h1": q_h1,
            "q_h2": q_h2,
            "q_ml": q_ml,
            "q_ml_basic": q_ml,
            "alpha_ml": alpha,
            "p_turb": p_turb,
            "q_ml_scaling": q_ml_scaling,
        }
        self.running = True
        self.schedule = BaseScheduler(self)
        # init environment and data collector
        self.init_env()
        self.init_dc()

    def get_config(self, param, *args):
        return self.conf[param]

    # necessary in order to satisfy data collector interface
    get_m = partialmethod(get_config, "m")
    get_n = partialmethod(get_config, "n")
    get_j = partialmethod(get_config, "j")
    get_p_1 = partialmethod(get_config, "p_1")
    get_p_2 = partialmethod(get_config, "p_2")
    get_p_3 = partialmethod(get_config, "p_3")
    get_q_h1 = partialmethod(get_config, "q_h1")
    get_q_h2 = partialmethod(get_config, "q_h2")
    get_q_ml = partialmethod(get_config, "q_ml")
    get_q_ml_basic = partialmethod(get_config, "q_ml_basic")
    get_alpha_ml = partialmethod(get_config, "alpha_ml")
    get_p_turb = partialmethod(get_config, "p_turb")
    get_q_ml_scaling = partialmethod(get_config, "q_ml_scaling")

    def get_time(self, *args):
        return int(self.schedule.time)

    def init_env(self):
        # init reality
        r = Reality("R1", self)
        self.schedule.add(r)
        # init humans
        for i in range(self.conf["n"]):
            h = Human(f"H{i+1}", self)
            self.schedule.add(h)
        # init ML agents
        # determine ML dimensions
        self.conf["ml_dims"] = random_dims(self.conf["m"], self.conf["j"])
        # init one agent per dimension
        for i in self.conf["ml_dims"]:
            m = MLAgent(f"ML{i+1}", self, i)
            self.schedule.add(m)
        # init organization
        o = OrganizationalCode("O1", self)
        self.schedule.add(o)
        return

    def init_dc(self):
        # data collector enables tracking of metric at each time step
        self.datacollector = DataCollector(
            model_reporters={
                "time": self.get_time,
                "m": self.get_m,
                "n": self.get_n,
                "j": self.get_j,
                "p_1": self.get_p_1,
                "p_2": self.get_p_2,
                "p_3": self.get_p_3,
                "q_h1": self.get_q_h1,
                "q_h2": self.get_q_h2,
                "q_ml": self.get_q_ml_basic,
                "alpha_ml": self.get_alpha_ml,
                "p_turb": self.get_p_turb,
                "q_ml_scaling": self.get_q_ml_scaling,
                "avg_q_ml": calc_avg_q_ml,
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
        return self.schedule.agents[-1]

    def get_human_agents(self):
        return self.schedule.agents[1:(1 + self.conf["n"])]

    def get_ml_agents(self):
        return self.schedule.agents[(1 + self.conf["n"]):-1]

    def update_kls(self):
        for h in self.get_human_agents():
            h.update_kl()
        for ml in self.get_ml_agents():
            ml.update_kl()
        code = self.get_org_code()
        code.update_kl()
        return

    def scale_q_ml(self):
        # for avg. human knowledge related manipulation
        scaling = self.conf["q_ml_scaling"]
        if scaling == "on":
            # for belief related manipulation
            exp_grp = self.get_exp_grp()
            ml_dims = self.conf["ml_dims"]
            q_ml = []
            for dim in ml_dims:
                # get knowledgeable group
                exp_grp_dim = list(
                    filter(lambda h: (h.state[dim] != 0), exp_grp))
                # get basic parameters
                reality = self.get_reality().state[dim]
                q_ml_basic = self.conf["q_ml_basic"]

                if len(exp_grp_dim) > 0:
                    # if expert group exists, count correct and incorrect beliefs
                    votes = [h.state[dim] for h in exp_grp_dim]
                    c = Counter(votes)

                    if len(c) > 1:
                        # if expert group has correct and incorrect beliefs calculate difference
                        k = c.get(reality)-c.get((-1)*reality)
                    else:
                        if votes[0] == reality:
                            # all expert beliefs are correct
                            k = c.get(reality)
                        else:
                            # all expert beliefs are incorrerect
                            k = c.get((-1)*reality)
                    # see Google Drive/Forschung/MISQ/ExtensionDesign for formulas
                    alpha = self.conf["alpha_ml"]
                    beta = math.log((1-q_ml_basic)/q_ml_basic)
                    q_ml.append(round(1/(1+math.e**(((-1)*k/alpha)+beta)), 3))
                else:
                    # if there are no experts, use basic q_ml value
                    q_ml.append(q_ml_basic)
            self.conf["q_ml"] = q_ml
        return

    def environmental_turbulence(self):
        reality = self.get_reality()
        reality.turbulence()
        return

    def step(self):
        try:
            # calculate knowledge levels
            self.update_kls()
            # determine expert group for this time step
            self.exp_grp = self.get_exp_grp()
            # scale q_ml according to human KL
            self.scale_q_ml()
            # update all agents
            self.schedule.step()
            # update reality according to turbulence
            self.environmental_turbulence()
            # collect metrics for this time step
            self.datacollector.collect(self)
        except Exception as e:
            # log potential erros, but continue with next iteration
            print("The following error occurred:")
            print(e)
            print("Model configuration:")
            print(self.conf)
