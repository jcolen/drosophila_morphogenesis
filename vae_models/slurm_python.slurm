#!/bin/bash
#
#SBATCH --job-name=INPUTS
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/INPUTS_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/INPUTS_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --mem=32G
#SBATCH --ntasks=4

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /project/vitelli/jonathan/REDO_fruitfly/src/vae_models
beta=0
python INPUTS_forecasting.py --beta $beta
