# fixed parameters
fixed_params = {
    "m": 30,
    "n": 50,
    "j": 15,
    "q_h1": 0.1,
    "q_h2": 0.5,
    "q_ml_scaling": "on",
    "q_d_scaling": "on",
}

# test configuration
test_params = {
    "p_1": [0.1, 0.9],
    "p_2": [0.1, 0.9],
    "p_3": [0.1, 0.9],
    "q_ml": [0.5, 0.9],
    "q_d": [0.5, 0.9],
    "alpha": [5, 10],
    "p_turb": [0, 0.1],
}

# full configuration
full_params = {
    "p_1": [0.1, 0.5, 0.9],
    "p_2": [0.1, 0.5, 0.9],
    "p_3": [0.1, 0.5, 0.9],
    "q_ml": [0.4, 0.9],
    "q_d": [0.4, 0.9],
    "alpha": [1, 5, 10, 25, 50],
    "p_turb": [0, 0.02, 0.1],
}
