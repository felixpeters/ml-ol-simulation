#!/bin/bash
# create environment
conda create -n misq_sim python=3.7
export PATH=/opt/conda/envs/misq_sim/bin:$PATH
source activate misq_sim

# set conda-forge as primary channel
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

# install packages
pip install mesa
conda install pandas notebook scipy numpy
