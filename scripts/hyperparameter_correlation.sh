#!/bin/bash
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=16   # number of cores per task
##SBATCH --gres=gpu:1        # number of GPUs (should match -n)
#SBATCH --nodelist=havoc    # if you need specific nodes
##SBATCH --exclude=como,manchester,blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes,atlas
#SBATCH -t 0-01:00          # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/Generalization_metrics_for_NLP/      # working directory
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
python hyperparameter_correlation.py \
--metric $1 \
--bleu_type $2 \
--group $3 \
--distribution $4 \
--fitting_method $5 \
--dataset $6 \
--calculate_or_plot $7 \
--model_size_param $8 $9 \

wait
date
