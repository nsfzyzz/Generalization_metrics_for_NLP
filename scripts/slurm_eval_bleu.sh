#!/bin/bash
#SBATCH --array=1-128
##SBATCH --array=1,10,20,22,31,41,43,52,62,64,73,83,85,94,104,106,115,125
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8   # number of cores per task
#SBATCH --gres=gpu:1        # number of GPUs (should match -n)
##SBATCH --nodelist=ace,manchester,bombe,como,pavia,luigi,zanino    # if you need specific nodes
##SBATCH --nodelist=ace,manchester,bombe
#SBATCH --exclude=blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes,atlas,zanino,luigi,como,pavia,ace,bombe
#SBATCH -t 2-00:00          # time requested (D-HH:MM)
#SBATCH -D /work/yyaoqing/Good_vs_bad_data/NLP_metrics_Simpson
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate pytorch-transformer
export PYTHONUNBUFFERED=1


cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/bleu_config.txt)
sample=$(echo $cfg | cut -f 1 -d ' ')
depth=$(echo $cfg | cut -f 2 -d ' ')
width=$(echo $cfg | cut -f 3 -d ' ')
lr=$(echo $cfg | cut -f 4 -d ' ')
dropout=$(echo $cfg | cut -f 5 -d ' ')
head=$(echo $cfg | cut -f 6 -d ' ')

CKPTPATH=/work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_sample"$sample"_depth"$depth"_width"$width"_lr"$lr"_dropout"$dropout"
echo $CKPTPATH
#mkdir $CKPTPATH

srun -N 1 -n 1 python eval_bleu_loss.py \
--checkpoint_dir $CKPTPATH \
--max_batches 200 \
--dataset WMT \
--num_epochs 20 \
--starting_epoch 1 \
--num-heads $head \
--embedding-dimension $width \
#--efficient-eval &

wait
date
