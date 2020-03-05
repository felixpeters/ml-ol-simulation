# ml-ol-simulation

This repository contains all code for running a simulation which examines the
effects of ML adoption on organizational learning. All models are based on
the agent-based model developed by [James March
(1991)](http://strategy.sjsu.edu/www.stable/pdf/March,%20J.%20G.%20%281991%29.%20Organization%20Science%202%281%29%2071-87.pdf)
and use the [Mesa framework](https://github.com/projectmesa/mesa/) (Python) as 
technical foundation.

## Repository structure

The following files are included in this repository:
- `utils/`: Utility functions for agent initialization, metric calculation and
data preprocessing
- `models/`: Here you can find the actual simulation models, separated into
agent and model definitions
- `nbs/`: Notebooks used for testing complex operations, e.g., efficient metric
calculations
- `data/`: Collected data from simulation runs will be saved here
- `/`: installing and running scripts, environment setup (Docker)

## Setup

All simulations are run in Docker containers in order to create a standardized
and reproducible environment. The following steps are needed in order to run
simulations:
1. Clone repository to target machine: `git clone
git@github.com:felixpeters/ml-ol-simulation.git`
2. Create `data` folder in repository root: `mkdir data` (on UNIX-based systems)
3. Install Docker if not already done: [installation
instructions](https://docs.docker.com/install/)

## Running simulations

The logic for running simulations is included in the running scripts (starting
with `run_`) in the project root. These scripts are meant to be executed inside
the Docker container and contain all necessary configuration (i.e., parameter
levels defining configurations, number of iterations for each configuration).

There are two ways of executing the running scripts:
1. (if [Make](https://www.gnu.org/software/make/) is installed) Run `make
run-image RUN_SCRIPT=run_original_march.py` (replace script name for executing
other models)
2. You have to build and run the Docker container manually:
    - Run `docker build -t felixpeters/ai-sim .` to build the image
    - Run ``docker run --rm -it --name ai-sim-runner -v `pwd`/data:/ai-sim/data
      felixpeters/ai-sim:latest /bin/bash -c "/bin/bash run.sh
      run_original_march.py"``

## Contact

Email: [peters@is.tu-darmstadt.de](mailto:peters@is.tu-darmstadt.de)
