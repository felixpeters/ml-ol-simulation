test_config = {
    "num_iterations": 1,
    "num_steps": 100,
    "fixed_params": {
        "m": 30,
        "n": 50,
        "j": 15,
        "q_h1": 0.1,
        "q_h2": 0.5,
        "q_ml_scaling": "on",
    },
    "variable_params": {
        "p_1": [0.1, 0.9],
        "p_2": [0.1, 0.9],
        "p_3": [0.1, 0.9],
        "q_ml": [0.5, 0.9],
        "alpha": [5, 50],
        "p_turb": [0, 0.1],
    },
}

run_config = {
    "num_iterations": 80,
    "num_steps": 200,
    "fixed_params": {
        "m": 30,
        "n": 50,
        "j": 15,
        "q_h1": 0.1,
        "q_h2": 0.5,
        "q_ml_scaling": "on",
    },
    "variable_params": {
        "p_1": [0.1, 0.5, 0.9],
        "p_2": [0.1, 0.5, 0.9],
        "p_3": [0.1, 0.5, 0.9],
        "q_ml": [0.2, 0.3, 0.5, 0.7, 0.8],
        "alpha": [1, 5, 10, 25, 50],
        "p_turb": [0, 0.02, 0.1],
    },
}
