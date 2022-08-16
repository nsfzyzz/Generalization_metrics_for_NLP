#!/bin/bash
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2   # number of cores per task
##SBATCH --gres=gpu:1        # number of GPUs (should match -n)
#SBATCH --nodelist=havoc    # if you need specific nodes
##SBATCH --exclude=blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes
#SBATCH -t 1-00:00          # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/Good_vs_bad_data/NLP_metrics_Simpson      # working directory
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww
source normalize_powerlaw.sh
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
python test_ww_collections.py \
$1 \
$2 \
--result-suffix "results_original_alpha.pkl" \
--width 512 \
--dataset $5 \
--num-samples $3 \
--num-layers $4 \
--mp-fit \
--randomize \
--distribution "power_law" \
--num-epochs $6

wait
date
