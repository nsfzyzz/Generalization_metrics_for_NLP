#!/bin/bash
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8   # number of cores per task
#SBATCH --gres=gpu:1        # number of GPUs (should match -n)
#SBATCH --nodelist=ace    # if you need specific nodes
##SBATCH --exclude=como,manchester,blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes,atlas
#SBATCH -t 1-00:00          # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/Good_vs_bad_data/NLP_metrics_Simpson      # working directory
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate pytorch-transformer
export PYTHONUNBUFFERED=1

srun -N 1 -n 1 python test_measures_collections.py \
$1 \
--result_suffix "robust_measures.pkl" \
--epochs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
--width 512 \
--dataset WMT \
--num-samples $2 \
--test_robust_measures \
--calculate_margin \
--calculate_pac_bayes \
--num-layers $3

wait
date
