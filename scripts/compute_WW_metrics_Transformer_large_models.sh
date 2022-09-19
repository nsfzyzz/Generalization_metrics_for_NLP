#!/bin/bash
#SBATCH -p rise             # partition (queue)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 1                # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=2   # number of cores per task
##SBATCH --gres=gpu:1        # number of GPUs (should match -n)
#SBATCH --nodelist=havoc    # if you need specific nodes
##SBATCH --exclude=blaze,flaminio,freddie,r[1-6,8-16],havoc,steropes
#SBATCH -t 2-00:00          # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/Good_vs_bad_data/NLP_metrics_Simpson      # working directory
#SBATCH -o slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate ww
source normalize_powerlaw.sh
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1

python test_ww_collections.py /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr0.25_dropout0.1 /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr0.25_dropout0.1 --result-suffix results.pkl --width 1024 --dataset WMT --num-samples 320000 --num-layers 6 --mp-fit --randomize --distribution TPL --num-epochs 20 --starting-epoch 20 --num-heads 16 &
python test_ww_collections.py /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr0.375_dropout0.1 /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr0.375_dropout0.1 --result-suffix results.pkl --width 1024 --dataset WMT --num-samples 320000 --num-layers 6 --mp-fit --randomize --distribution TPL --num-epochs 20 --starting-epoch 20 --num-heads 16 &
python test_ww_collections.py /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr0.5_dropout0.1 /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr0.5_dropout0.1 --result-suffix results.pkl --width 1024 --dataset WMT --num-samples 320000 --num-layers 6 --mp-fit --randomize --distribution TPL --num-epochs 20 --starting-epoch 20 --num-heads 16 &
python test_ww_collections.py /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr0.75_dropout0.1 /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr0.75_dropout0.1 --result-suffix results.pkl --width 1024 --dataset WMT --num-samples 320000 --num-layers 6 --mp-fit --randomize --distribution TPL --num-epochs 20 --starting-epoch 20 --num-heads 16 &
python test_ww_collections.py /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr1.0_dropout0.1 /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr1.0_dropout0.1 --result-suffix results.pkl --width 1024 --dataset WMT --num-samples 320000 --num-layers 6 --mp-fit --randomize --distribution TPL --num-epochs 20 --starting-epoch 20 --num-heads 16 &
python test_ww_collections.py /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr1.5_dropout0.1 /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_large_dimension_debug_sample320000_depth6_lr1.5_dropout0.1 --result-suffix results.pkl --width 1024 --dataset WMT --num-samples 320000 --num-layers 6 --mp-fit --randomize --distribution TPL --num-epochs 20 --starting-epoch 20 --num-heads 16 &

wait
date
