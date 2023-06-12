#!/bin/bash

betas=("1e0" "1e1" "1e2" "1e3" "1e4")
betas=("1e-4" "1e-3" "1e-2" "1e-1")
betas=("1e3" "1e4")
betas=("0")
betas=("1e-3" "1e-2" "1e-1" "1e0" "1e1" "1e2")
for beta in ${betas[@]}; do
	echo $beta
	cp slurm_python.slurm job.slurm
	sed -i "s/BB/${beta}/" job.slurm
	sbatch job.slurm
done

rm job.slurm
