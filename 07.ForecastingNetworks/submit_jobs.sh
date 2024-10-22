#!/bin/bash

inputs=("gene" "myosin" "cadherin" "gene_combo" "protein_combo" "protein_gene")
inputs=("gene_combo")
#inputs=("myosin" "cadherin" "protein_combo")
#inputs=("protein_combo")
#inputs=("myosin")
for input in ${inputs[@]}; do
	echo $input
	cp slurm_python.slurm job.slurm
	sed -i "s/INPUTS/${input}/" job.slurm
	sbatch job.slurm
done

rm job.slurm
