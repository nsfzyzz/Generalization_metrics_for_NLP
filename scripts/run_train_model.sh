#!/bin/bash

# Note: For now, we only train IWSLT checkpoints with dropout 0.1

#for samples in 40000 80000 120000 160000 200000
#do
#    for depth in 4 5 6 7 8
#    do
#        for lr in 0.5 0.75 1.0 1.5 2.0
#        do
#        # order is [num_samples] [depth] [lr_factor] [dropout]
#        sbatch scripts/train_model.sh $samples $depth $lr 0.1
#        done
#    done
#done

# Training experiments for WMT14

for samples in 160000 320000 640000 1280000
do
    for depth in 4 5 6 7 8
    do
    for lr in 0.625 0.375
        do
        # order is [num_samples] [depth] [lr_factor] [dropout]
        #mkdir /work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_sample"$samples"_depth"$depth"_lr"$lr"_dropout0.1 
        sbatch scripts/train_model.sh $samples $depth $lr 0.1
        done
    done
done
