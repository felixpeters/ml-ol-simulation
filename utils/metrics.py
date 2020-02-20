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
    return model.schedule.agents[1].kl

def calc_human_kl(model):
    humans = model.human_agents(active_only=True)
    kls = [h.kl for h in humans]
    return np.mean(kls)

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

