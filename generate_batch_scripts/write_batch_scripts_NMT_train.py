# This file is used to generate the training files from the txt files
import argparse

def get_slurm_script(code, args):
    
    slurm_script = f'''#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=8 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH --nodelist={args.node} # if you need specific nodes
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
export PATH="/data/yyaoqing/anaconda3/bin:$PATH"
conda activate pytorch-transformer
export WANDB_PROJECT={args.wandb_name}
export PYTHONUNBUFFERED=1

{code}

wait
date
'''
    return slurm_script

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-name", type=str, help="wandb name", default='NMT_TRAIN')
    parser.add_argument("--node", type=str, help="rise slurm node", default='bombe')
    parser.add_argument("--folder-suffix", type=str, default="")

    args = parser.parse_args()
    
    with open('generate_batch_scripts/commands.txt', 'r') as f:
        lines = f.readlines()
        file_id = 0
        for line in lines:
            tmp_script = get_slurm_script(line, args)
            with open(f'{args.wandb_name}{args.folder_suffix}_{file_id}.sh', 'w') as fwrite:
                fwrite.write(tmp_script)
            
            file_id += 1

    with open('submission.sh', 'w') as fsubmit:

        for file_id in range(file_id):
            fsubmit.write(f'sbatch {args.wandb_name}{args.folder_suffix}_{file_id}.sh\n')
