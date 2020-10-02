simulation_config = {
    "test": {
        "num_iterations": 2,
        "num_steps": 500,
        "fixed_params": {
            "m": 30,
            "n": 50,
            "j": 15,
            "q_h1": 0.1,
            "q_h2": 0.5,
            "q_ml_scaling": "on",
        },
        "slice": {
            "slice1": {
                "q_ml": 0.5,
                "alpha": 5,
            },
            "slice2": {
                "q_ml": 0.5,
                "alpha": 50,
            },
        },
        "variable_params": {
            "p_1": [0.1, 0.9],
            "p_2": [0.1, 0.9],
            "p_3": [0.1, 0.9],
            "p_turb": [0, 0.1],
        },
    },
    "no_ml": {
        "num_iterations": 80,
        "num_steps": 500,
        "fixed_params": {
            "m": 30,
            "n": 50,
            "j": 0,
            "q_h1": 0.1,
            "q_h2": 0.5,
            "q_ml_scaling": "off",
            "q_ml": 0.5,
            "alpha": 5,
        },
        "slice": {
            "slice1": {
                "p_turb": 0,
            },
            "slice2": {
                "p_turb": 0.02,
            },
            "slice3": {
                "p_turb": 0.2,
            },
        },
        "variable_params": {
            "p_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "p_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "p_3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
    "default": {
        "num_iterations": 80,
        "num_steps": 500,
        "fixed_params": {
            "m": 30,
            "n": 50,
            "j": 15,
            "q_h1": 0.1,
            "q_h2": 0.5,
            "q_ml_scaling": "on",
        },
        "slice": {
            "slice1": {
                "q_ml": 0.2,
                "alpha": 1,
                "p_turb": 0,
            },
            "slice2": {
                "q_ml": 0.2,
                "alpha": 1,
                "p_turb": 0.02,
            },
            "slice3": {
                "q_ml": 0.2,
                "alpha": 1,
                "p_turb": 0.2,
            },
            "slice4": {
                "q_ml": 0.2,
                "alpha": 10,
                "p_turb": 0,
            },
            "slice5": {
                "q_ml": 0.2,
                "alpha": 10,
                "p_turb": 0.02,
            },
            "slice6": {
                "q_ml": 0.2,
                "alpha": 10,
                "p_turb": 0.2,
            },
            "slice7": {
                "q_ml": 0.2,
                "alpha": 50,
                "p_turb": 0,
            },
            "slice8": {
                "q_ml": 0.2,
                "alpha": 50,
                "p_turb": 0.02,
            },
            "slice9": {
                "q_ml": 0.2,
                "alpha": 50,
                "p_turb": 0.2,
            },
            "slice10": {
                "q_ml": 0.5,
                "alpha": 1,
                "p_turb": 0,
            },
            "slice11": {
                "q_ml": 0.5,
                "alpha": 1,
                "p_turb": 0.02,
            },
            "slice12": {
                "q_ml": 0.5,
                "alpha": 1,
                "p_turb": 0.2,
            },
            "slice13": {
                "q_ml": 0.5,
                "alpha": 10,
                "p_turb": 0,
            },
            "slice14": {
                "q_ml": 0.5,
                "alpha": 10,
                "p_turb": 0.02,
            },
            "slice15": {
                "q_ml": 0.5,
                "alpha": 10,
                "p_turb": 0.2,
            },
            "slice16": {
                "q_ml": 0.5,
                "alpha": 50,
                "p_turb": 0,
            },
            "slice17": {
                "q_ml": 0.5,
                "alpha": 50,
                "p_turb": 0.02,
            },
            "slice18": {
                "q_ml": 0.5,
                "alpha": 50,
                "p_turb": 0.2,
            },
            "slice19": {
                "q_ml": 0.8,
                "alpha": 1,
                "p_turb": 0,
            },
            "slice20": {
                "q_ml": 0.8,
                "alpha": 1,
                "p_turb": 0.02,
            },
            "slice21": {
                "q_ml": 0.8,
                "alpha": 1,
                "p_turb": 0.2,
            },
            "slice22": {
                "q_ml": 0.8,
                "alpha": 10,
                "p_turb": 0,
            },
            "slice23": {
                "q_ml": 0.8,
                "alpha": 10,
                "p_turb": 0.02,
            },
            "slice24": {
                "q_ml": 0.8,
                "alpha": 10,
                "p_turb": 0.2,
            },
            "slice25": {
                "q_ml": 0.8,
                "alpha": 50,
                "p_turb": 0,
            },
            "slice26": {
                "q_ml": 0.8,
                "alpha": 50,
                "p_turb": 0.02,
            },
            "slice27": {
                "q_ml": 0.8,
                "alpha": 50,
                "p_turb": 0.2,
            },
        },
        "variable_params": {
            "p_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "p_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "p_3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    },
}
