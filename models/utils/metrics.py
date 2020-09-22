import numpy as np


def calc_kl(reality, beliefs):
    equals = 0
    for i in range(len(reality)):
        if reality[i] == beliefs[i]:
            equals += 1
    res = float(equals) / float(len(reality))
    return res


def calc_code_kl(model):
    code = model.get_org_code()
    return code.kl


def calc_human_kl(model):
    humans = model.get_human_agents()
    return np.mean([h.kl for h in humans])


def calc_dissim(model):
    n = model.conf["n"]
    num_dims = model.conf["m"]
    humans = model.get_human_agents()
    beliefs = np.vstack([h.state for h in humans])
    rows, cols = np.triu_indices(n, 1)
    comp_sum = np.sum(beliefs[rows] != beliefs[cols])
    coeff = 2 / (num_dims * n * (n - 1))
    return coeff * comp_sum


def calc_kl_var(model):
    humans = model.get_human_agents()
    kls = [h.kl for h in humans]
    return np.var(kls)


def calc_avg_q_d(model):
    result = 0.0
    if model.conf["q_d_scaling"] == "on":
        result = np.mean(model.conf["q_d"])
    else:
        result = model.conf["q_d"]
    return result


def calc_avg_q_ml(model):
    result = 0.0
    if model.conf["q_ml_scaling"] == "on":
        result = np.mean(model.conf["q_ml"])
    else:
        result = model.conf["q_ml"]
    return result
