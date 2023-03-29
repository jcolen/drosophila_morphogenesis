#!/bin/bash

inputs=("gene" "protein" "gene_combo" "protein_combo" "protein_gene")
inputs=("protein_combo")
#inputs=("gene_combo")
#inputs=("protein")
inputs=("gene" "gene_combo" "protein_gene")
inputs=("protein")
for input in ${inputs[@]}; do
	echo $input
	cp slurm_python.slurm job.slurm
	sed -i "s/INPUTS/${input}/" job.slurm
	sbatch job.slurm
done

rm job.slurm
