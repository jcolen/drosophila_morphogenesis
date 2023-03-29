#!/bin/bash
#
#SBATCH --job-name=run_ClosedLoopPINN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/run_ClosedLoopPINN_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/run_ClosedLoopPINN_%A.out
#SBATCH --partition=vitelli-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294,midway3-[0277-0281]
#SBATCH --ntasks=1
#SBATCH --mem=4G

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /project/vitelli/jonathan/REDO_fruitfly/src/ClosedLoopPINN
python run_ClosedLoopPINN.py
