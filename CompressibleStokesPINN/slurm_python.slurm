#!/bin/bash
#
#SBATCH --job-name=run_CompressibleStokesPINN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/run_CompressibleStokesPINN_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/run_CompressibleStokesPINN_%A.out
#SBATCH --partition=vitelli-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --ntasks=1
#SBATCH --mem=4G

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /project/vitelli/jonathan/REDO_fruitfly/src/CompressibleStokesPINN
python run_CompressibleStokesPINN.py --beta 1e5
