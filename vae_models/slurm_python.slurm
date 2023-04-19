#!/bin/bash
#
#SBATCH --job-name=INPUTS
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/INPUTS_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/INPUTS_%A.out
#SBATCH --partition=vitelli-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --mem=48G
#SBATCH --ntasks=4

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /project/vitelli/jonathan/REDO_fruitfly/src/vae_models
beta=0
echo "STARTING EMBRYO SPLIT"
python INPUTS_forecasting.py --beta $beta --mode embryo

echo "STARTING LR SPLIT"
python INPUTS_forecasting.py --beta $beta --mode LR

echo "STARTING AP SPLIT"
python INPUTS_forecasting.py --beta $beta --mode AP
