#!/bin/bash

# Note: For now, we only have WMT checkpoints with dropout 0.1

#for samples in 160000 320000 640000 1280000 2560000
for samples in 160000 320000 640000
do
    #for depth in 4 5 6 7 8
    for depth in 6
    do
        #for lr in 0.0625 0.125 0.25 0.375 0.5 0.625 0.75 1.0 1.5 2.0
        for lr in 0.0625
        do
        # order is [checkpoint dir] [result_dir] [num_samples] [depth]
        #sbatch scripts/compute_ww_metric.sh
        sbatch scripts/compute_ww_metric_PL.sh \
        "/work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_sample${samples}_depth${depth}_lr${lr}_dropout0.1" \
        "/work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_sample${samples}_depth${depth}_lr${lr}_dropout0.1" \
        $samples $depth WMT 40
        done
    done
done
