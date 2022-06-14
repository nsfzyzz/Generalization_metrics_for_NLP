#!/bin/bash

# Note: For now, we only have IWSLT checkpoints with dropout 0.1

for samples in 40000 80000 120000 160000 200000
do
    for depth in 4 5 6 7 8
    do
        for lr in 0.5 0.75 1.0 1.5 2.0
        do
        # order is [checkpoint dir] [result_dir] [num_samples] [depth]
        sbatch scripts/compute_ww_metric.sh \
        "/work/rkunani/pytorch-transformer/checkpoint/IWSLT_sample${samples}_depth${depth}_lr${lr}_dropout0.1" \
        "/work/rkunani/pytorch-transformer/checkpoint/IWSLT_sample${samples}_depth${depth}_lr${lr}_dropout0.1" \
        $samples $depth IWSLT
        done
    done
done