#!/bin/bash

# Note: For now, we only have WMT checkpoints with dropout 0.1

#for samples in 40000 80000 120000 160000 200000
for samples in 640000
do
    for depth in 4 5 6 7 8
    #for depth in 4
    do
        for lr in 0.5 0.75 1.0 1.5 2.0
        #for lr in 0.5
        do
        # order is [checkpoint dir] [result_dir] [num_samples] [depth]
        sbatch scripts/compute_ww_metric.sh \
	"/work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_sample${samples}_depth${depth}_lr${lr}_dropout0.1" \
        "/work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson/WMT14_sample${samples}_depth${depth}_lr${lr}_dropout0.1" \
        $samples $depth WMT
        done
    done
done
