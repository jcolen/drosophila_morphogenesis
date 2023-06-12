#!/bin/bash
#
#SBATCH --job-name=run_MyosinDynamicsPINN
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/run_MyosinDynamicsPINN_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/run_MyosinDynamicsPINN_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --ntasks=1
#SBATCH --mem=4G

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /project/vitelli/jonathan/REDO_fruitfly/src/MyosinDynamicsPINN
python run_MyosinDynamicsPINN.py --beta BB
