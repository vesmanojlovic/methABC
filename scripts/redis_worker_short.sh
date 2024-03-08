#!/bin/bash

# slurm settings
#SBATCH -D /users/adby616/methABC
#SBATCH --partition=nodes
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=48
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH -o /users/adby616/archive/abc-temp/%x_%j.o
#SBATCH -e /users/adby616/archive/abc-temp/%x_%j.e

# prepare environment, e.g. set path

# run
abc-redis-worker --host=10.10.0.21 --port=2666 --runtime=3d --processes=384

