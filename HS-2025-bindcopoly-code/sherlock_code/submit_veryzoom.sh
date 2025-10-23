#!/bin/bash
#
#SBATCH --job-name=vryzoom
#
#SBATCH --time=400:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH -p normal

module load system ruse

module load python/3.6.1

module load py-numpy/1.18.1_py36

module load py-scipy/1.10.1_py39

ruse python3 multipros_phase_diag_starmap_veryzoom.py
