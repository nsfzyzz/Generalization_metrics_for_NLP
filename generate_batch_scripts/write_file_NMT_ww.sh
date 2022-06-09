# width experiment normal
#python write_script.py --dataset IWSLT --training-type normal --experiment-type width --command-file width_exp_normal.sh
#python write_script.py --dataset IWSLT --training-type const_lr --experiment-type width --command-file width_exp_const.sh
#chmod +x width_exp_const.sh
#python write_script.py --dataset IWSLT --training-type const_lr --experiment-type width --command-file width_exp2_const.sh --second-exp
#chmod +x width_exp2_const.sh

# sample experiment
#python write_script.py --dataset IWSLT --training-type normal --experiment-type sample --command-file sample_exp_IWSLT.sh
#chmod +x sample_exp_IWSLT.sh
#python write_script.py --dataset WMT --training-type normal --experiment-type sample --command-file sample_exp_WMT.sh
#chmod +x sample_exp_WMT.sh
# IWSLT sample experiment with larger models
#python write_script.py --dataset IWSLT --training-type normal --mp-fit --randomize --experiment-type sample --command-file sample_exp_IWSLT.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_IWSLT.sh
# WMT sample experiment with larger models
#python write_script.py --dataset WMT --training-type normal --mp-fit --randomize --experiment-type sample --command-file sample_exp_WMT.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_WMT.sh


# IWSLT sample experiment with larger models, training for only 20 epochs
# normal training with dropout
# python write_script.py --dataset IWSLT --training-type normal --only20 --mp-fit --randomize --experiment-type sample --command-file sample_exp_IWSLT_new.sh --IWSLT-width 512 --mkdir
# without dropout
#python write_script.py --dataset IWSLT --training-type no_dropout --only20 --mp-fit --randomize --experiment-type sample --command-file sample_exp_IWSLT_new.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_IWSLT_new.sh
# WMT sample experiment with larger models, training for only 20 epochs
# normal training with dropout
# python write_script.py --dataset WMT --training-type normal --only20 --mp-fit --randomize --experiment-type sample --command-file sample_exp_WMT_new.sh --IWSLT-width 512 --mkdir
# without dropout
#python write_script.py --dataset WMT --training-type no_dropout --only20 --mp-fit --randomize --experiment-type sample --command-file sample_exp_WMT_new.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_WMT_new.sh


# sample experiment
# apply different measures
#python write_script.py --script-type measures --dataset IWSLT --training-type normal --experiment-type sample --command-file sample_exp_IWSLT_measures.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_IWSLT_measures.sh
#python write_script.py --script-type measures --dataset WMT --training-type normal --experiment-type sample --command-file sample_exp_WMT_measures.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_WMT_measures.sh

# lr experiment
#python write_script.py --dataset IWSLT --training-type normal --experiment-type lr --command-file lr_exp.sh
#chmod +x lr_exp.sh


# IWSLT sample experiment with the normalization fixed
# without dropout
#python write_script.py --dataset IWSLT --metric-folder metrics_new --training-type no_dropout --only20 --mp-fit --randomize --experiment-type sample --command-file sample_exp_IWSLT_normalization.sh --IWSLT-width 512 --mkdir
# with dropout
#python write_script.py --dataset IWSLT --metric-folder metrics_new --training-type normal --only20 --mp-fit --randomize --experiment-type sample --command-file sample_exp_IWSLT_normalization.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_IWSLT_normalization.sh


# apply different measures after changing to 20 epochs
#python write_script.py --script-type measures --dataset IWSLT --training-type normal --only20 --experiment-type sample --command-file sample_exp_IWSLT_measures_new.sh --IWSLT-width 512 --mkdir
#python write_script.py --script-type measures --dataset IWSLT --training-type no_dropout --only20 --experiment-type sample --command-file sample_exp_IWSLT_measures_new.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_IWSLT_measures_new.sh
#python write_script.py --script-type measures --dataset WMT --training-type normal --only20 --experiment-type sample --command-file sample_exp_WMT_measures.sh --IWSLT-width 512 --mkdir
#python write_script.py --script-type measures --dataset WMT --training-type no_dropout --only20 --experiment-type sample --command-file sample_exp_WMT_measures.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_WMT_measures.sh

# IWSLT sample experiment with the exponential cutoff
# without dropout
#python write_script.py --dataset IWSLT --metric-folder metrics_expcutoff --training-type no_dropout --only20 --mp-fit --randomize --experiment-type sample --exp-cutoff --command-file sample_exp_IWSLT_expcutoff.sh --IWSLT-width 512 --mkdir
# with dropout
#python write_script.py --dataset IWSLT --metric-folder metrics_expcutoff --training-type normal --only20 --mp-fit --randomize --experiment-type sample --exp-cutoff --command-file sample_exp_IWSLT_expcutoff.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_IWSLT_expcutoff.sh


# IWSLT sample experiment with the exponential cutoff and lambda can be negative
# This is hard because negative lambda leads to some problems
# without dropout
#python write_script.py --dataset IWSLT --metric-folder metrics_expcutoff_positive_lambda --training-type no_dropout --only20 --mp-fit --randomize --experiment-type sample --exp-cutoff --negative-lambda --command-file sample_exp_IWSLT_expcutoff_positive_lambda.sh --IWSLT-width 512 --mkdir
# with dropout
#python write_script.py --dataset IWSLT --metric-folder metrics_expcutoff_positive_lambda --training-type normal --only20 --mp-fit --randomize --experiment-type sample --exp-cutoff --negative-lambda --command-file sample_exp_IWSLT_expcutoff_positive_lambda.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_IWSLT_expcutoff_positive_lambda.sh

# WMT sample experiment with the exponential cutoff
# without dropout
#python generate_batch_scripts/write_script.py --dataset WMT --metric-folder metrics_expcutoff --training-type no_dropout --only20 --mp-fit --randomize --experiment-type sample --exp-cutoff --command-file sample_exp_WMT_expcutoff.sh --IWSLT-width 512 --mkdir
# with dropout
#python generate_batch_scripts/write_script.py --dataset WMT --metric-folder metrics_expcutoff --training-type normal --only20 --mp-fit --randomize --experiment-type sample --exp-cutoff --command-file sample_exp_WMT_expcutoff.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_WMT_expcutoff.sh

# WMT sample experiment by fixing finger
# without dropout
#python write_script.py --dataset WMT --metric-folder metrics_new --training-type no_dropout --only20 --mp-fit --randomize --experiment-type sample --fix-finger --command-file sample_exp_WMT_fix_finger.sh --IWSLT-width 512 --mkdir
# with dropout
#python write_script.py --dataset WMT --metric-folder metrics_new --training-type normal --only20 --mp-fit --randomize --experiment-type sample --fix-finger --command-file sample_exp_WMT_fix_finger.sh --IWSLT-width 512 --mkdir
#chmod +x sample_exp_WMT_fix_finger.sh

# test all measures after including margin and pac-bayes into the calculation
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset IWSLT --training-type normal --only20 --experiment-type sample --command-file sample_exp_IWSLT_measures_normal.sh --IWSLT-width 512 --mkdir --result-suffix robust_measures_include_margin.pkl --wandb-project-name NMT_MEASURES_MARGIN --python-buffer --calculate-margin --calculate-pac-bayes
#chmod +x sample_exp_IWSLT_measures_normal.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset IWSLT --training-type no_dropout --only20 --experiment-type sample --command-file sample_exp_IWSLT_measures_no_dropout.sh --IWSLT-width 512 --mkdir --result-suffix robust_measures_include_margin.pkl --wandb-project-name NMT_MEASURES_MARGIN --python-buffer --calculate-margin --calculate-pac-bayes
#chmod +x sample_exp_IWSLT_measures_no_dropout.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset WMT --training-type normal --only20 --experiment-type sample --command-file sample_exp_WMT_measures_normal.sh --IWSLT-width 512 --mkdir --result-suffix robust_measures_include_margin.pkl --wandb-project-name NMT_MEASURES_MARGIN --python-buffer --calculate-margin --calculate-pac-bayes
#chmod +x sample_exp_WMT_measures_normal.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset WMT --training-type no_dropout --only20 --experiment-type sample --command-file sample_exp_WMT_measures_no_dropout.sh --IWSLT-width 512 --mkdir --result-suffix robust_measures_include_margin.pkl --wandb-project-name NMT_MEASURES_MARGIN --python-buffer --calculate-margin --calculate-pac-bayes
#chmod +x sample_exp_WMT_measures_no_dropout.sh

# test the margin only and see if it is always negative (there is now a new measure called "true_margin")
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset IWSLT --training-type normal --only20 --experiment-type sample --command-file sample_exp_IWSLT_measures_normal_true_margin.sh --IWSLT-width 512 --mkdir --result-suffix robust_measures_true_margin.pkl --wandb-project-name NMT_MEASURES_TRUE_MARGIN --python-buffer --calculate-margin --eval-suffix _true_margin
#chmod +x sample_exp_IWSLT_measures_normal_true_margin.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset IWSLT --training-type no_dropout --only20 --experiment-type sample --command-file sample_exp_IWSLT_measures_no_dropout_true_margin.sh --IWSLT-width 512 --mkdir --result-suffix robust_measures_true_margin.pkl --wandb-project-name NMT_MEASURES_TRUE_MARGIN --python-buffer --calculate-margin --eval-suffix _true_margin
#chmod +x sample_exp_IWSLT_measures_no_dropout_true_margin.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset WMT --training-type normal --only20 --experiment-type sample --command-file sample_exp_WMT_measures_normal_true_margin.sh --IWSLT-width 512 --mkdir --result-suffix robust_measures_true_margin.pkl --wandb-project-name NMT_MEASURES_TRUE_MARGIN --python-buffer --calculate-margin --eval-suffix _true_margin
#chmod +x sample_exp_WMT_measures_normal_true_margin.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset WMT --training-type no_dropout --only20 --experiment-type sample --command-file sample_exp_WMT_measures_no_dropout_true_margin.sh --IWSLT-width 512 --mkdir --result-suffix robust_measures_true_margin.pkl --wandb-project-name NMT_MEASURES_TRUE_MARGIN --python-buffer --calculate-margin --eval-suffix _true_margin
#chmod +x sample_exp_WMT_measures_no_dropout_true_margin.sh

# test the BLEU score generalization
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset IWSLT --training-type normal --only20 --experiment-type sample --command-file sample_exp_IWSLT_bleu_normal.sh --IWSLT-width 512 --mkdir --result-suffix bleu.pkl --wandb-project-name NMT_MEASURES_BLEU --python-buffer --eval-suffix _bleu --not-test-robust-measures --not-retrieve-ww --test-bleu
#chmod +x sample_exp_IWSLT_bleu_normal.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset IWSLT --training-type no_dropout --only20 --experiment-type sample --command-file sample_exp_IWSLT_bleu_no_dropout.sh --IWSLT-width 512 --mkdir --result-suffix bleu.pkl --wandb-project-name NMT_MEASURES_BLEU --python-buffer --eval-suffix _bleu --not-test-robust-measures --not-retrieve-ww --test-bleu
#chmod +x sample_exp_IWSLT_bleu_no_dropout.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset WMT --training-type normal --only20 --experiment-type sample --command-file sample_exp_WMT_bleu_normal.sh --IWSLT-width 512 --mkdir --result-suffix bleu.pkl --wandb-project-name NMT_MEASURES_BLEU --python-buffer --eval-suffix _bleu --not-test-robust-measures --not-retrieve-ww --test-bleu
#chmod +x sample_exp_WMT_bleu_normal.sh
#python generate_batch_scripts/write_script.py --directory-depth 0 --script-type measures --dataset WMT --training-type no_dropout --only20 --experiment-type sample --command-file sample_exp_WMT_bleu_no_dropout.sh --IWSLT-width 512 --mkdir --result-suffix bleu.pkl --wandb-project-name NMT_MEASURES_BLEU --python-buffer --eval-suffix _bleu --not-test-robust-measures --not-retrieve-ww --test-bleu
#chmod +x sample_exp_WMT_bleu_no_dropout.sh

## Train the sample experiment
# experiment_type="sample"
# training_types=("no_dropout" "normal")
# datasets=("IWSLT" "WMT")

# for training_type in "${training_types[@]}"
# do
#     for dataset in "${datasets[@]}"
#     do
#         python generate_batch_scripts/write_script.py --directory-depth 0 --script-type train --dataset $dataset \
#         --training-type $training_type --experiment-type $experiment_type \
#         --command-file "$experiment_type"_exp_"$dataset"_"$training_type"_training.sh \
#         --IWSLT-width 512 --mkdir --folder-suffix _exp3 #_exp2

#         chmod +x "$experiment_type"_exp_"$dataset"_"$training_type"_training.sh
#     done
# done

# cat "$experiment_type"_exp_*_*_training.sh > commands.txt
# rm "$experiment_type"_exp_*_*_training.sh
# mv commands.txt ./generate_batch_scripts/
# python generate_batch_scripts/write_batch_scripts_NMT_train.py

## Train the LR experiment
# experiment_type="lr"
# training_types=("no_dropout" "normal")
# datasets=("IWSLT" "WMT")

# for training_type in "${training_types[@]}"
# do
#     for dataset in "${datasets[@]}"
#     do
#         python generate_batch_scripts/write_script.py --directory-depth 0 --script-type train --dataset $dataset \
#         --training-type $training_type --experiment-type $experiment_type \
#         --command-file "$experiment_type"_exp_"$dataset"_"$training_type"_training.sh \
#         --IWSLT-width 512 --exclude-standard --folder-suffix _exp2 --mkdir 

#         chmod +x "$experiment_type"_exp_"$dataset"_"$training_type"_training.sh
#     done
# done

# cat "$experiment_type"_exp_*_*_training.sh > commands.txt
# rm "$experiment_type"_exp_*_*_training.sh
# mv commands.txt ./generate_batch_scripts/
# python generate_batch_scripts/write_batch_scripts_NMT_train.py --node bombe --wandb-name NMT_train_"$experiment_type"


## Train the depth experiment
experiment_type="depth"
training_types=("no_dropout" "normal")
datasets=("IWSLT" "WMT")

for training_type in "${training_types[@]}"
do
    for dataset in "${datasets[@]}"
    do
        python generate_batch_scripts/write_script.py --directory-depth 0 --script-type train --dataset $dataset \
        --training-type $training_type --experiment-type $experiment_type \
        --command-file "$experiment_type"_exp_"$dataset"_"$training_type"_training.sh \
        --IWSLT-width 512 --exclude-standard --folder-suffix _exp2  #--mkdir

        chmod +x "$experiment_type"_exp_"$dataset"_"$training_type"_training.sh
    done
done

cat "$experiment_type"_exp_*_*_training.sh > commands.txt
rm "$experiment_type"_exp_*_*_training.sh
mv commands.txt ./generate_batch_scripts/
python generate_batch_scripts/write_batch_scripts_NMT_train.py --node bombe --wandb-name NMT_train_$experiment_type


## Train the width experiment
# The standard experiment is here (width=512)
# So we should not use --exclude-standard
# experiment_type="width"
# training_types=("no_dropout" "normal")
# datasets=("IWSLT" "WMT")

# for training_type in "${training_types[@]}"
# do
#     for dataset in "${datasets[@]}"
#     do
#         python generate_batch_scripts/write_script.py --directory-depth 0 --script-type train --dataset $dataset \
#         --training-type $training_type --experiment-type $experiment_type \
#         --command-file "$experiment_type"_exp_"$dataset"_"$training_type"_training.sh \
#         --folder-suffix _exp2 # --mkdir

#         chmod +x "$experiment_type"_exp_"$dataset"_"$training_type"_training.sh
#     done
# done

# cat "$experiment_type"_exp_*_*_training.sh > commands.txt
# rm "$experiment_type"_exp_*_*_training.sh
# mv commands.txt ./generate_batch_scripts/
# python generate_batch_scripts/write_batch_scripts_NMT_train.py --node bombe --wandb-name NMT_train_$experiment_type

