#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8 # number of cores per task
##SBATCH --gres=gpu:1
#SBATCH --nodelist=manchester # if you need specific nodes
##SBATCH --exclude=zanino,ace,blaze,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes,blaze
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/Good_vs_bad_data/NLP_metrics
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
export PYTHONUNBUFFERED=1

tar -czvf ckpt_new.tar.gz ../checkpoint/NMT_epochs/*/w512/*lr_factor*/
#scp ckpt_new.tar.gz ubuntu@ec2-3-233-9-172.compute-1.amazonaws.com:~/Good_vs_bad_data/

wait
date