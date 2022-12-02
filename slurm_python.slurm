#!/bin/bash
#
#SBATCH --job-name=translation_vae
#SBATCH --output=/scratch/midway3/jcolen/sbatch_outputs/translation_vae_%A.out
#SBATCH --error=/scratch/midway3/jcolen/sbatch_outputs/translation_vae_%A.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294,midway3-[0277-0281]
#SBATCH --ntasks=2

echo "Starting"

source /home/jcolen/.bashrc
module load python
conda activate /project/vitelli/ml_venv

cd /project/vitelli/jonathan/REDO_fruitfly/src
#python atlas_processing/build_derivative_library.py
python translation_vae.py
