#!/bin/bash
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8   # number of cores per task
#SBATCH --gres=gpu:1        # number of GPUs (should match -n)
#SBATCH --nodelist=ace      # if you need specific nodes
##SBATCH --exclude=manchester,como,pavia,luigi,zanino,steropes,atlas,blaze,flaminio,freddie,r[1-6,8-16],havoc
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

srun -N 1 -n 1 python eval_bleu_loss.py \
--checkpoint_dir $1 \
--max_batches $2 \
--dataset $3

wait
date
