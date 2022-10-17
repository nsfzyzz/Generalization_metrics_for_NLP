'''
This file is used to generate the directories of all experiments.
'''

import argparse

WMT_sample_list = [160000, 320000, 640000, 1280000, 2560000]
IWSLT_sample_list = [40000, 80000, 120000, 160000, 200000]
depth_list = [4, 5, 6, 7, 8]
lr_list = ["0.0625", "0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "1.0"]
width_list = [256, 384, 512, 768, 1024]
head_list = [4, 6, 8, 12, 16]


def generate_WMT_depth_experiments(fwrite):
    ## Generate sample x learning rate x depth grid on WMT
    for sample in WMT_sample_list:
        for depth in depth_list:
            for lr in lr_list:
                fwrite.write("    os.path.join(CKPT_DIR, " + f"\"WMT14_sample{sample}_depth{depth}_width512_lr{lr}_dropout0.1\"" + "),\n")


def generate_IWSLT_depth_experiments(fwrite):
    ## Generate sample x learning rate x depth grid on IWSLT
    for sample in IWSLT_sample_list:
        for depth in depth_list:
            for lr in lr_list:
                fwrite.write("    os.path.join(CKPT_DIR, " + f"\"IWSLT_sample{sample}_depth{depth}_width512_lr{lr}_dropout0.1\"" + "),\n")


def generate_WMT_width_experiments(fwrite):
    ## Generate sample x learning rate x width grid on WMT
    for sample in WMT_sample_list:
        for width, head in zip(width_list, head_list):
            for lr in lr_list:
                fwrite.write("    os.path.join(CKPT_DIR, " + f"\"WMT14_sample{sample}_depth6_width{width}_lr{lr}_dropout0.1\"" + "),\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--CKPT_DIR", type=str, default = '/data/yyaoqing/Generalization_metrics_for_NLP/checkpoint', 
                        help="path to save all the checkpoints")
    args = parser.parse_args()

    ## First, generate experiments for varying different hyperparameters
    starting_text = """"""
    starting_text += "import os\n\n"
    starting_text += f"CKPT_DIR = \"{args.CKPT_DIR}\"\n\n"
    starting_text += "EXPERIMENTS = {\n"

    with open('experiments_hyperparameters.py', 'w') as fwrite:
        fwrite.write(starting_text)
        fwrite.write('''    "IWSLT_depth": [\n''')
        generate_IWSLT_depth_experiments(fwrite)
        fwrite.write('''],
    
"WMT14_depth": [\n''')
        generate_WMT_depth_experiments(fwrite)
        fwrite.write('''],

"WMT14_width": [\n''')
        generate_WMT_width_experiments(fwrite)
        fwrite.write(''']}''')

    ## Then, generate experiments for time-wise correlations
    starting_text = """"""
    starting_text += "import os\n\n"
    starting_text += f"CKPT_DIR = \"{args.CKPT_DIR}\"\n\n"
    starting_text += "EXPERIMENTS = {\n"

    with open('experiments_time_wise.py', 'w') as fwrite:
        fwrite.write(starting_text)
        fwrite.write('''    "IWSLT": [\n''')
        generate_IWSLT_depth_experiments(fwrite)
        fwrite.write('''],
    
"WMT": [\n''')
        generate_WMT_width_experiments(fwrite)
        generate_WMT_depth_experiments(fwrite)
        fwrite.write(''']}''')

