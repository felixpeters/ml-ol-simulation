#!/bin/bash
#SBATCH -J ml-ol-sim
#SBATCH --mail-user=peters@is.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH -e ~/%x.err.%j
#SBATCH -o ~/%x.out.%j
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --exclusive


echo "Job $SLURM_JOB_ID started at $(date)"
module purge
module load gcc
module load python
echo "Loaded required gcc and python modules"
pip install -r requirements.txt
echo "Installed necessary Python modules"
python run.py --config default --slice qml08alpha50
EXITSTATUS=$?
echo "Job $SLURM_JOB_ID finished at $(date)"
exit $EXITSTATUS
