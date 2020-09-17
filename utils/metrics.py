import numpy as np

def calc_kl(reality, beliefs):
    equals = 0
    for i in range(len(reality)):
        if reality[i] == beliefs[i]: equals += 1
    res = float(equals) / float(len(reality))
    return res

def track_model_steps(model):
    return model.datacollector

def calc_code_kl(model):
    code = model.get_org_code()
    return code.kl

def calc_human_kl(model):
    humans = model.get_human_agents()
    return np.mean([h.kl for h in humans])

def calc_ds_kl(model):
    num_hum = model.conf["num_regular"]
    num_ds = model.conf["num_data_scientist"]
    ds = model.schedule.agents[(2 + num_hum):(2 + num_hum + num_ds)]
    return np.mean([d.kl for d in ds])

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

def calc_dissim(model):
    num_humans = model.conf["num_humans"]
    num_dims = model.conf["belief_dims"]
    humans = model.get_human_agents()
    beliefs = np.vstack([h.state for h in humans])
    rows, cols = np.triu_indices(num_humans, 1)
    comp_sum = np.sum(beliefs[rows] != beliefs[cols])
    coeff = 2 / (num_dims * num_humans * (num_humans - 1))
    return coeff * comp_sum

def calc_kl_var(model):
    humans = model.get_human_agents()
    kls = [h.kl for h in humans]
    return np.var(kls)

def calc_data_qual(model):
    data = model.get_data()
    return data.kl
