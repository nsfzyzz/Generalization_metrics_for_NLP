import argparse
import os
import sys
os.system('pwd')
sys.path.insert(1, '../')
sys.path.insert(1, './')
from utils.utils_NMT import get_experiment_folders_and_epochs

def main(args):

    ckpt_folders, ckpt_epochs, widths, samples, lr_factors, depths = get_experiment_folders_and_epochs(args)
    assert len(ckpt_folders) == len(ckpt_epochs)
    commands = []

    if args.wandb_project_name != "":
        commands.append(f'export WANDB_PROJECT={args.wandb_project_name}')
    if args.python_buffer:
        commands.append('export PYTHONUNBUFFERED=1')
    
    for ckpt_folder, epochs, width, sample, lr_factor, depth in zip(ckpt_folders, ckpt_epochs, widths, samples, lr_factors, depths):
        for epoch in epochs:

            add_command = True # This is used to check if the result file exists

            ckpt_file = f'{ckpt_folder}/net_epoch_{epoch}.ckpt'
            print(ckpt_file)

            if not args.script_type == 'train':
                assert os.path.exists(ckpt_file)
                assert os.path.exists(os.path.join(ckpt_folder, args.metric_folder))
                metric_epoch_folder = os.path.join(ckpt_folder, args.metric_folder, f'epoch_{epoch}')
                if not os.path.exists(metric_epoch_folder):
                    os.mkdir(metric_epoch_folder)                
            
            command_suffix = ''

            if args.script_type == 'WW':
                if args.mp_fit:
                    command_suffix += ' --mp-fit'
                if args.randomize:
                    command_suffix += ' --randomize'
                if args.fix_finger:
                    command_suffix += ' --fix-finger'
                    command_suffix += f' --continuous-estimate {args.continuous_estimate}'
                    command_suffix += f' --result-suffix results_Charles_fix_finger.pkl'

                if depth:
                    command_suffix += f' --num-layers {depth}'

                command_suffix += f' --distribution {args.distribution}'
                
                command = f"python test_ww_collections.py {ckpt_file}" \
                            + f" {metric_epoch_folder} --width {width}" \
                            + f" --dataset {args.dataset} --num-samples {sample}" \
                            + f" --save-plot{command_suffix} 1>{ckpt_folder}/{args.metric_folder}/epoch_{epoch}/ww.log" \
                            + f" 2>{ckpt_folder}/{args.metric_folder}/epoch_{epoch}/ww.err"

                # TODO: sometimes the result file is not results.pkl. 
                # In that case, one should replace this name
                ww_result_file = os.path.join(ckpt_folder, args.metric_folder, f"epoch_{epoch}", 'results.pkl')
                if os.path.exists(ww_result_file):
                    add_command = False
                    print(ww_result_file + " exists already!")

            elif args.script_type == 'train':
                if sample!=0:
                    command_suffix += f' --subsampling --num-samples {sample}'
                if lr_factor:
                    command_suffix += f' --lr-factor {lr_factor}'
                if depth:
                    command_suffix += f' --num-layers {depth}'
                
                if args.dataset == 'WMT':
                    dataset_name = 'WMT14'
                    command_suffix += ' --max-gradient-steps 800000'
                elif args.dataset == 'IWSLT':
                    dataset_name = 'IWSLT'
                else:
                    raise ValueError('dataset not implemented.')

                if args.training_type == 'normal':
                    command_suffix += ' --dropout 0.1'

                command = f"python training_script.py --batch_size 1500 --dataset_name {dataset_name}" \
                            + f" --language_direction G2E --checkpoint-path {ckpt_folder}" \
                            + f" --checkpoint_freq 1 --num_of_epochs 20 --embedding-dimension {width}" \
                            + f" --lr-inverse-dim" \
                            + f"{command_suffix} 1>{ckpt_folder}/log_0.txt" \
                            + f" 2>{ckpt_folder}/err_0.txt"

            elif args.script_type == 'measures':

                if args.result_suffix != "":
                    command_suffix += ' --result_suffix ' + args.result_suffix
                if args.calculate_margin:
                    command_suffix += ' --calculate_margin'
                if args.calculate_pac_bayes:
                    command_suffix += ' --calculate_pac_bayes'

                if depth:
                    command_suffix += f' --num-layers {depth}'

                if not args.not_test_robust_measures:
                    command_suffix += ' --test_robust_measures'
                if not args.not_retrieve_ww:
                    command_suffix += ' --retrieve_ww_measures'
                if args.test_bleu:
                    command_suffix += ' --test_bleu'
                    bleu_result_file = os.path.join(ckpt_folder, 'bleu.pkl')
                    if os.path.exists(bleu_result_file):
                        add_command = False
                        print(bleu_result_file + " exists already!")
                else:
                    if args.result_suffix:
                        measure_result_file = os.path.join(ckpt_folder, args.result_suffix)
                    else:
                        measure_result_file = os.path.join(ckpt_folder, 'robust_measures.pkl')
                    if os.path.exists(measure_result_file):
                        add_command = False
                        print(measure_result_file + " exists already!")

                epochs_string = " ".join([str(x) for x in epochs])
                command = f"python test_measures_collections.py {ckpt_folder}" \
                            + f" --width {width} --epochs {epochs_string}" \
                            + f" --dataset {args.dataset} --num-samples {sample}" \
                            + f"{command_suffix} 1>{ckpt_folder}/eval_measures{args.eval_suffix}.log" \
                            + f" 2>{ckpt_folder}/eval_measures{args.eval_suffix}.err"
            
            elif args.script_type == 'SVDSmoothing':
                
                if sample!=0:
                    # This means that the experiment uses subsampling
                    command_suffix += ' --subsampling'

                command_suffix += ' --SVDSmoothing-tail-num'
                command = f"python test_SVDsmoothing_collections.py {ckpt_file}" \
                            + f" {metric_epoch_folder} --width {width}" \
                            + f" --dataset {args.dataset} --num-samples {sample}" \
                            + f" {command_suffix} 1>{ckpt_folder}/{args.metric_folder}/epoch_{epoch}/SVDsmoothing.log" \
                            + f" 2>{ckpt_folder}/{args.metric_folder}/epoch_{epoch}/SVDsmoothing.err"

            if add_command: # Only add the command if the results do not exist
                commands.append(command)
            if args.script_type == 'measures' or args.script_type == 'train':
                break

    with open(args.command_file, 'w') as f:
        for command in commands:
            f.write(command+'\n')

    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", choices=['IWSLT', 'WMT'], default='IWSLT')
    parser.add_argument("--training-type", type=str, choices=['const_lr', 'normal', 'no_dropout'], default='normal')
    parser.add_argument("--experiment-type", type=str, choices=['width', 'sample', 'lr', 'depth'], default='width')
    parser.add_argument("--command-file", type=str, default='file1.txt')
    parser.add_argument("--IWSLT-width", type=int, default=512)
    parser.add_argument("--mkdir", action='store_true', default=False)
    parser.add_argument("--mp-fit", action='store_true', help="fitting the model using MP Fit.")
    parser.add_argument("--randomize", action='store_true', help="use randomized matrix to check correlation trap.")
    parser.add_argument("--metric-folder", type=str, default='metrics')
    parser.add_argument("--distribution", type=str, default="power_law")
    parser.add_argument("--result-suffix", type=str, default="")
    parser.add_argument("--eval-suffix", type=str, default="")
    parser.add_argument("--fix-finger", action='store_true', default=False)
    parser.add_argument("--not-test-robust-measures", action='store_true', default=False)
    parser.add_argument("--not-retrieve-ww", action='store_true', default=False)
    parser.add_argument("--test-bleu", action='store_true', default=False)
    parser.add_argument("--python-buffer", action='store_true', default=False)
    parser.add_argument("--calculate-margin", action='store_true', default=False)
    parser.add_argument("--calculate-pac-bayes", action='store_true', default=False)
    parser.add_argument("--continuous-estimate", type=float, default=30)
    parser.add_argument("--directory-depth", type=int, default=0)
    parser.add_argument("--wandb-project-name", type=str, default="")
    parser.add_argument("--folder-suffix", type=str, default="")
    parser.add_argument("--exclude-standard", action='store_true')

    parser.add_argument("--script-type", type=str, choices=['WW', 'SVDSmoothing', "measures", 'train'], 
                                    default='WW', help="which type of experiments")

    args = parser.parse_args()

    main(args)