#!/bin/bash
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2   # number of cores per task
##SBATCH --gres=gpu:1        # number of GPUs (should match -n)
#SBATCH --nodelist=havoc    # if you need specific nodes
##SBATCH --exclude=blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes
#SBATCH -t 1-00:00          # time requested (D-HH:MM)
#SBATCH -D /home/eecs/rkunani/NLP_metrics      # working directory
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate pytorch-transformer
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
python test_ww_collections.py \
$1 \
$2 \
--result-suffix "ww_results.pkl" \
--width 512 \
--dataset $5 \
--num-samples $3 \
--num-layers $4 \
--mp-fit \
--randomize \
--distribution "truncated_power_law"

wait
date