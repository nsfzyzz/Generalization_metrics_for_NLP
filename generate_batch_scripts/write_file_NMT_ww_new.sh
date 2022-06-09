# Generate bash files to fit four distributions

distributions=("lognormal" "exponential" "truncated_power_law" "power_law")
folder_names=("_lognormal" "_exponential" "_expcutoff" "")
training_types=("no_dropout" "normal")
datasets=("IWSLT" "WMT")


for i in ${!distributions[@]}
do
    distribution=${distributions[$i]}
    folder=${folder_names[$i]}

    for training_type in "${training_types[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            python generate_batch_scripts/write_script.py --dataset $dataset --metric-folder metrics$folder \
            --training-type $training_type --mp-fit --randomize --experiment-type sample --distribution $distribution \
            --command-file sample_exp_"$dataset"_"$distribution"_"$training_type".sh --IWSLT-width 512 --mkdir \
            --folder-suffix _exp3 # _exp2
            chmod +x sample_exp_"$dataset"_"$distribution"_"$training_type".sh
        done
    done
done