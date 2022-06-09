#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 2 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8 # number of cores per task
#SBATCH --gres=gpu:2
#SBATCH --nodelist=manchester # if you need specific nodes
##SBATCH --exclude=ace,blaze,bombe,flaminio,freddie,luigi,pavia,r[10,16],atlas,como,havoc,steropes
#SBATCH -t 7-00:00 # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/Good_vs_bad_data/NLP_metrics
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
export PATH="/data/yyaoqing/anaconda3/bin:$PATH"
conda activate pytorch-transformer
export WANDB_PROJECT=WW
export PYTHONUNBUFFERED=1


mkdir ../checkpoint/MLM/pretrain_BERT_lr_2e-5_wd_0.01_save_steps_500_max_steps_18000_randlayer_2
srun -N 1 -n 1 --gres=gpu:1 python language_modeling.py --ckpt_folder ../checkpoint/MLM/pretrain_BERT_lr_2e-5_wd_0.01_save_steps_500_max_steps_18000_randlayer_2 --pretrain --save_steps 500 --max_steps 18000 --weight_decay 0.01 --lr 2e-5 --eval --eval_ww --model_checkpoint bert-base-uncased --randomize_layers_num 2 1>../checkpoint/MLM/pretrain_BERT_lr_2e-5_wd_0.01_save_steps_500_max_steps_18000_randlayer_2/train.log 2>../checkpoint/MLM/pretrain_BERT_lr_2e-5_wd_0.01_save_steps_500_max_steps_18000_randlayer_2/train.err &
mkdir ../checkpoint/MLM/pretrain_BERT_lr_2e-5_wd_0.01_save_steps_500_max_steps_18000_randlayer_3
srun -N 1 -n 1 --gres=gpu:1 python language_modeling.py --ckpt_folder ../checkpoint/MLM/pretrain_BERT_lr_2e-5_wd_0.01_save_steps_500_max_steps_18000_randlayer_3 --pretrain --save_steps 500 --max_steps 18000 --weight_decay 0.01 --lr 2e-5 --eval --eval_ww --model_checkpoint bert-base-uncased --randomize_layers_num 3 1>../checkpoint/MLM/pretrain_BERT_lr_2e-5_wd_0.01_save_steps_500_max_steps_18000_randlayer_3/train.log 2>../checkpoint/MLM/pretrain_BERT_lr_2e-5_wd_0.01_save_steps_500_max_steps_18000_randlayer_3/train.err &

wait
date


