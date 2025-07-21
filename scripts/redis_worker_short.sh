#!/bin/bash

# slurm settings
#SBATCH -D /users/adby616/methABC
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH -o /users/adby616/archive/abc-temp/%x_%j.o
#SBATCH -e /users/adby616/archive/abc-temp/%x_%j.e

# run
srun -n 48 abc-redis-worker --host=10.10.0.21 --port=2166 --runtime=3d --processes=2

