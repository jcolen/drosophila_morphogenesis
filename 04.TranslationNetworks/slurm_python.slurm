#!/bin/bash
#
#SBATCH --job-name=wt_translation
#SBATCH --output=/scratch/midway3/jcolen/wt_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-vitelli
#SBATCH --exclude=midway3-0294
#SBATCH --array=0-2
#SBATCH --mem=16GB
#SBATCH --ntasks=4

echo "Starting"

source /home/jcolen/.bashrc

module load python
conda deactivate
conda deactivate
conda deactivate
conda deactivate
conda deactivate
conda activate /project/vitelli/ml_venv

cd /project/vitelli/jonathan/REDO_fruitfly/revision/04.TranslationNetworks

betas=(1e-3 1e-4 1e-5)
beta=${betas[${SLURM_ARRAY_TASK_ID}]}

echo "STARTING BETA " ${beta}

epochs=500
#python train_myosin_translation.py --beta $beta --epochs $epochs
python train_myosin_translation.py --beta $beta --epochs $epochs --use_pmg_cf_mask --logdir ./tb_logs/pmg_cf_mask
