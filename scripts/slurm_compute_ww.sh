#!/bin/bash
#SBATCH --array=1-200
#SBATCH -p xxxxx             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2   # number of cores per task
#SBATCH --nodelist=xxxxx
##SBATCH --exclude=xxxxx
#SBATCH -t 2-00:00          # time requested (D-HH:MM)
#SBATCH -D /data/user-name/Generalization_metrics_for_NLP/
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate NLP_metrics
source normalize_powerlaw.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

#distribution=truncated_power_law
#result_file=results.pkl
#config_file=ww_tpl_config

#distribution=exponential
#result_file=results_exponential.pkl
#config_file=ww_exponential_config

distribution=power_law
result_file=results_original_alpha.pkl
config_file=ww_pl_config

cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/"$config_file".txt)
sample=$(echo $cfg | cut -f 1 -d ' ')
depth=$(echo $cfg | cut -f 2 -d ' ')
width=$(echo $cfg | cut -f 3 -d ' ')
lr=$(echo $cfg | cut -f 4 -d ' ')
dropout=$(echo $cfg | cut -f 5 -d ' ')
head=$(echo $cfg | cut -f 6 -d ' ')

CKPTPATH=/data/xxxxx/Generalization_metrics_for_NLP/checkpoint/WMT14_sample"$sample"_depth"$depth"_width"$width"_lr"$lr"_dropout"$dropout"
echo $CKPTPATH
#mkdir $CKPTPATH

python compute_ww.py \
$CKPTPATH $CKPTPATH \
--result-suffix $result_file \
--width $width \
--dataset WMT \
--num-samples $sample \
--num-layers $depth \
--mp-fit --randomize \
--distribution $distribution \
--num-epochs 20 \
--starting-epoch 1 \
--num-heads $head &

wait
date
