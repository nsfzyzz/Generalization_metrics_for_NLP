#!/bin/bash
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2   # number of cores per task
##SBATCH --gres=gpu:1        # number of GPUs (should match -n)
#SBATCH --nodelist=havoc    # if you need specific nodes
##SBATCH --exclude=como,manchester,blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes,atlas
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
python plot_scatterplot.py \
--metric $1 \
--bleu_type $2 \
--group $3

wait
date