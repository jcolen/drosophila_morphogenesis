#!/bin/bash

inputs=("gene" "protein" "gene_combo" "protein_combo" "protein_gene")

for input in ${inputs[@]}; do
	echo $input
	cp slurm_python.slurm job.slurm
	sed -i "s/INPUTS/${input}/" job.slurm
	sbatch job.slurm
done

rm job.slurm
