#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=10 # number of cores per task
#SBATCH --nodelist=bombe # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/Good_vs_bad_data/checkpoint
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc

tar -czvf ALBERT.tar.gz */pretrain_ALBERT_lr_2e-5_wd_0.01_save_steps_200_max_steps_10000_randlayer_*/

wait
date