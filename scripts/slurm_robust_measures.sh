#!/bin/bash
#SBATCH --array=1-200
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8   # number of cores per task
#SBATCH --gres=gpu:1        # number of GPUs (should match -n)
##SBATCH --nodelist=ace,manchester,bombe,como,pavia,luigi,zanino    # if you need specific nodes
#SBATCH --exclude=blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes,atlas,zanino,luigi,como
#SBATCH -t 2-00:00          # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/Generalization_metrics_for_NLP/
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate NLP_metrics
export PYTHONUNBUFFERED=1


cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/robust_config.txt)
sample=$(echo $cfg | cut -f 1 -d ' ')
depth=$(echo $cfg | cut -f 2 -d ' ')
width=$(echo $cfg | cut -f 3 -d ' ')
lr=$(echo $cfg | cut -f 4 -d ' ')
dropout=$(echo $cfg | cut -f 5 -d ' ')
head=$(echo $cfg | cut -f 6 -d ' ')

CKPTPATH=/data/yyaoqing/Generalization_metrics_for_NLP/checkpoint/WMT14_sample"$sample"_depth"$depth"_width"$width"_lr"$lr"_dropout"$dropout"
echo $CKPTPATH
#mkdir $CKPTPATH

srun -N 1 -n 1 python test_measures_collections.py \
$CKPTPATH \
--result_suffix "robust_measures.pkl" \
--num-epochs 20 \
--width $width \
--dataset WMT \
--num-samples $sample \
--calculate_margin \
--calculate_pac_bayes \
--num-layers $depth \
--num-heads $head &

wait
date
