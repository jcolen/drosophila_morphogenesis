#!/bin/bash
#
#SBATCH --job-name=forecasting_vae
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/forecasting_vae_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/forecasting_vae_%A.out
#SBATCH --partition=vitelli-gpu
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
#beta=1e-6
#beta=1e-3
beta=0
python forecasting_vae.py --beta $beta
