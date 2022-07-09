#!/bin/bash

# Note: For now, we only train IWSLT checkpoints with dropout 0.1

#for samples in 40000 80000 120000 160000 200000
#do
#    for depth in 4 5 6 7 8
#    do
#        for lr in 0.5 0.75 1.0 1.5 2.0
#        do
        # order is [checkpoint dir] [num_samples] [depth]
#        sbatch scripts/compute_metric.sh \
#        "/work/rkunani/pytorch-transformer/checkpoint/IWSLT_sample${samples}_depth${depth}_lr${lr}_dropout0.1" \
#        $samples $depth
#        done
#    done
#done

# WMT experiment

for samples in 1280000 
do
    for depth in 4 5 6 7 8
    do
        for lr in 0.375 0.625
        #for lr in 0.5 0.75 1.0 1.5 2.0
        do
        # order is [checkpoint dir] [num_samples] [depth]
        sbatch scripts/compute_metric.sh \
        "/work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_sample${samples}_depth${depth}_lr${lr}_dropout0.1" \
        $samples $depth
        done
    done
done