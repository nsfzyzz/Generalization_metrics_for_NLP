import os

###################################
# Would you like to make directory?
###################################
#mkdir_command = "--mkdir"
mkdir_command = ""

#####################################
# Which type of environment variable?
#####################################
experiment_type = "sample"
#experiment_type = "lr"
# experiment_type = "depth"
# experiment_type = "width"

###########################
# Which type of experiment?
###########################
#script_type = "train"
#script_type = "measures"
#script_type = "bleu"
script_type = "WW"

##########################################
# Which nodes do we use for the three runs
##########################################
node_list = ["pavia", "ace", "manchester"]

if script_type == "WW":
    distributions=["lognormal", "exponential", "truncated_power_law", "power_law"]
    distribution_folder_names=["_lognormal", "_exponential", "_expcutoff", ""]
else:
    distributions=[""]
    distribution_folder_names=[""]

training_types=["no_dropout", "normal"]
datasets=["IWSLT", "WMT"]

python_command = "python generate_batch_scripts/write_script.py --directory-depth 0"
# TODO: The width experiment should use a different width
width = 512

script_type_combined = script_type
if script_type == "bleu":
    script_type_combined = "measures"
command_suffix = ""
if experiment_type in ["lr", "depth"]:
    command_suffix += " --exclude-standard"
if experiment_type != "width":
    command_suffix += f" --IWSLT-width {width}"
if script_type == "measures":
    command_suffix += " --result-suffix robust_measures_include_margin.pkl" \
        + " --calculate-margin --calculate-pac-bayes --not-retrieve-ww"
if script_type == "bleu":
    command_suffix += " --result-suffix bleu.pkl --eval-suffix _bleu" \
        + " --not-test-robust-measures --not-retrieve-ww --test-bleu"

folder_suffix_arg_list = ["", "--folder-suffix", "--folder-suffix"]
folder_suffix_list = ["", "_exp2", "_exp3"]
#folder_suffix_arg_list = ["--folder-suffix", "--folder-suffix"]
#folder_suffix_list = ["_exp2", "_exp3"]

for distribution, distribution_folder in zip(distributions, distribution_folder_names):
    if script_type == "WW":
        command_suffix += f" --metric-folder metrics{distribution_folder} --distribution {distribution}"
        if distribution == "power_law":
            command_suffix += " --mp-fit --randomize"

    for folder_suffix_arg, folder_suffix, node in zip(folder_suffix_arg_list, folder_suffix_list, node_list):
        for training_type in training_types:
            for dataset in datasets:
                if script_type == "WW":
                    command_file = f"{experiment_type}_exp_{dataset}_{training_type}_{script_type}_distribution_{distribution}{folder_suffix}.sh"
                else:
                    command_file = f"{experiment_type}_exp_{dataset}_{training_type}_{script_type}{folder_suffix}.sh"

                command = python_command + f" --script-type {script_type_combined} --dataset {dataset}" \
                    + f" --training-type {training_type} --experiment-type {experiment_type}" \
                    + f" --command-file {command_file}" \
                    + f" {mkdir_command} {command_suffix} {folder_suffix_arg} {folder_suffix}"

                os.system(command)
                os.system(f"chmod +x {command_file}")

        if script_type in ["measures", "train", "bleu"]:
            os.system(f"cat {experiment_type}_exp_*_*_{script_type}{folder_suffix}.sh > commands.txt")
            os.system(f"rm {experiment_type}_exp_*_*_{script_type}{folder_suffix}.sh")
            os.system("mv commands.txt ./generate_batch_scripts/")

            command = f"python generate_batch_scripts/write_batch_scripts_NMT_train.py" \
                        + f" --node {node}" \
                        + f" {folder_suffix_arg} {folder_suffix}" \
                        + f" --wandb-name NMT_{script_type}_{experiment_type}"
            os.system(command)
