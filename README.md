# ml-ol-simulation

This repository contains all code for running a simulation which examines the
effects of ML adoption on organizational learning (OL). All models are based on
the agent-based model developed by [James March
(1991)](http://strategy.sjsu.edu/www.stable/pdf/March,%20J.%20G.%20%281991%29.%20Organization%20Science%202%281%29%2071-87.pdf)
and use the [Mesa framework](https://github.com/projectmesa/mesa/) (Python) as
technical foundation.

## Repository structure

The following files are included in this repository:

- `utils/`: Utility functions for batch runners, configuration and data preprocessing
- `models/`: Actual simulation models (separated into agent and model definitions), along with utility functions for agent initialization and metric calculation
- `nbs/`: Notebooks used for testing model implementations, speedups (e.g., matrix-based metric calculations)
- `data/`: Collected data from simulation runs will be saved here
- `/`: Run script and environment configuration

## Setup

All simulations should be run in a virtual environment in order to create a standardized and reproducible environment.

The following assumptions are made about your system:

- Working installation of [Git](https://git-scm.com/)
- Working installation of [Python](https://www.python.org/downloads/) (at least version 3.7)
- Working shell for running command-line instructions (e.g., Git Bash, VS Code or PyCharm terminal)

The following setup steps have to be completed in order to run simulations:

1. Clone repository to target machine: `git clone git@github.com:felixpeters/ml-ol-simulation.git` or `git clone https://github.com/felixpeters/ml-ol-simulation.git`
2. Navigate to repository directory: `cd ml-ol-simulation` (on UNIX-based systems)
3. Create `data` folder in repository root: `mkdir data` (on UNIX-based systems)
4. Setup a virtual environment: `python -m venv venv`
5. Activate virtual environment: `source venv/bin/activate` (on UNIX-based systems)
6. Install required packages: `pip install -r requirements.txt`

## Running simulations

The logic for running simulations is included in the run script (`run.py`) in the project root. The script is meant to be run in the virtual environment.
Model runs can be configured using constants (see line starting `# define constants`). In detail, the following parameters can be set:

- `MODEL_NAME`: Determine the model to be run, currently `base` or `alt`
- `DATA_PATH`: Leave this untouched if you followed the setup instructions above
- `CPU_COUNT`: Set this to a fixed number if you wish, otherwise all available CPU cores will be used
- `CONFIG`: Determines which parameter set to use for simulation run, see `utils/config.py` for details

Conduct the following steps to run the simulation:

1. Activate the virtual environment (if it isn't already): `source venv/bin/activate`
2. Run the simulation: `python run.py`

The runtime can be estimated from the output of the run script (iteration speed is updated continuously).

## Contact

Email: [peters@is.tu-darmstadt.de](mailto:peters@is.tu-darmstadt.de)
