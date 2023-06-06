#!/bin/bash

betas=("1e1" "1e2" "1e3" "1e4")
betas=("1e5" "1e6" "1e7" "1e8")
for beta in ${betas[@]}; do
	echo $beta
	cp slurm_python.slurm job.slurm
	sed -i "s/BB/${beta}/" job.slurm
	sbatch job.slurm
done

rm job.slurm
