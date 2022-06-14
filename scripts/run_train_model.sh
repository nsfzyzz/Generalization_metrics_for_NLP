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

for samples in 160000
do
    #for depth in 6
    for depth in 4 5 7 8
    do
        for lr in 0.5 0.75 1.0 1.5 2.0
        do
        # order is [num_samples] [depth] [lr_factor] [dropout]
        sbatch scripts/train_model.sh $samples $depth $lr 0.1
        done
    done
done
