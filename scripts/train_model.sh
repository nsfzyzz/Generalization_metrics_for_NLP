#!/bin/bash
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8   # number of cores per task
#SBATCH --gres=gpu:1        # number of GPUs (should match -n)
##SBATCH --nodelist=ace,manchester,bombe,como,pavia,luigi,zanino    # if you need specific nodes
#SBATCH --exclude=blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes,atlas
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

srun -N 1 -n 1 python training_script.py \
--num_of_epochs 20 \
--dataset_name IWSLT \
--language_direction G2E \
--subsampling --num-samples $1 \
--num-layers $2 \
--lr-inverse-dim \
--lr-factor $3 \
--dropout $4 \
--checkpoint-path /work/rkunani/pytorch-transformer/checkpoint/IWSLT_sample$1_depth$2_lr$3_dropout$4

wait
date