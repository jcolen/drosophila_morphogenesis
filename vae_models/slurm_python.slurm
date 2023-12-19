#!/bin/bash
#
#SBATCH --job-name=INPUTS
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/INPUTS_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/INPUTS_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294,midway3-[0277-0281]
#SBATCH --mem=48G
#SBATCH --ntasks=4

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /project/vitelli/jonathan/REDO_fruitfly/release/vae_models


#Extended training for closed loop integration
#beta=1e-4
#epochs=500

#Training all models
beta=0
epochs=100

echo "STARTING EMBRYO SPLIT"
python INPUTS_forecasting.py --beta $beta --mode embryo --epochs $epochs
