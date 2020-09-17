#!/bin/bash
RUN_SCRIPT=$1
source /etc/profile.d/conda.sh
conda activate misq_sim
python $RUN_SCRIPT
