#!/bin/bash
#SBATCH --array=1-200
#SBATCH -p xxxxx             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8   # number of cores per task
#SBATCH --gres=gpu:1        # number of GPUs (should match -n)
##SBATCH --nodelist=xxxxx    # if you need specific nodes
#SBATCH --exclude=xxxxx
#SBATCH -t 7-00:00          # time requested (D-HH:MM)
#SBATCH -D /data/user-name/Generalization_metrics_for_NLP/
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate NLP_metrics
export PYTHONUNBUFFERED=1

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/train_config.txt)
sample=$(echo $cfg | cut -f 1 -d ' ')
depth=$(echo $cfg | cut -f 2 -d ' ')
width=$(echo $cfg | cut -f 3 -d ' ')
lr=$(echo $cfg | cut -f 4 -d ' ')
dropout=$(echo $cfg | cut -f 5 -d ' ')
head=$(echo $cfg | cut -f 6 -d ' ')

CKPTPATH=/data/user-name/Generalization_metrics_for_NLP/checkpoint/WMT14_sample"$sample"_depth"$depth"_width"$width"_lr"$lr"_dropout"$dropout"
echo $CKPTPATH
#mkdir $CKPTPATH

srun -N 1 -n 1 python training_script.py \
--num_of_epochs 20 \
--dataset_name WMT14 \
--language_direction G2E \
--subsampling --num-samples $sample \
--embedding-dimension $width \
--num-heads $head \
--num-layers $depth \
--lr-inverse-dim \
--lr-factor $lr \
--max-gradient-steps 100000000 \
--dropout $dropout \
--checkpoint-path $CKPTPATH \
1>$CKPTPATH/log_0.txt \
2>$CKPTPATH/err_0.txt &

wait
date
