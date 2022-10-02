#!/bin/bash

# There are three hyperparameter grids, two for WMT14 and one for IWSLT
datasets=("WMT14" "IWSLT" "WMT14")
size_params=("depth" "depth" "width")

# For each grid, we calculate both the bleu scores and the generalization gap (train bleu - test bleu)
bleu_types=("id_bleu" "id_bleu_gap")

# We test both with and without normalizing the metric by the number of samples
for adjust_measure in '--adjust_measures_back' ''
do
    for i in ${!datasets[@]}; do
        dataset=${datasets[$i]}
        size_param=${size_params[$i]}

        for fitting_method in 'ODR'
        do
        
            for bleu_type in ${bleu_types[@]}
            do
                for group in 'sample'
                do

                    # This is for PL
                    for metric in 'PL_alpha' 'PL_KS_distance' 'rand_distance' 'mp_softrank' 'log_norm' 'log_spectral_norm' 'L2' 'L2_DIST' 'PARAM_NORM' 'FRO_DIST' 'LOG_SUM_OF_FRO' 'DIST_SPEC_INIT' 'LOG_PROD_OF_FRO' 'LOG_SUM_OF_SPEC' 'LOG_PROD_OF_SPEC' 'PATH_NORM' 'stable_rank' 'tail_mean_vec_entropy' 'bulk_mean_vec_entropy' 'entropy' 'alpha_weighted' 'log_alpha_norm' 'INVERSE_MARGIN' 'LOG_PROD_OF_SPEC_OVER_MARGIN' 'LOG_SUM_OF_SPEC_OVER_MARGIN' 'LOG_PROD_OF_FRO_OVER_MARGIN' 'LOG_SUM_OF_FRO_OVER_MARGIN' 'PATH_NORM_OVER_MARGIN' 'PACBAYES_INIT' 'PACBAYES_ORIG' 'PACBAYES_FLATNESS' 'PACBAYES_MAG_INIT' 'PACBAYES_MAG_ORIG' 'PACBAYES_MAG_FLATNESS' 'W_CKA' 'FRO_OVER_SPEC' 'LOG_SPEC_INIT_MAIN' 'LOG_SPEC_ORIG_MAIN' 'LOG_SPEC_ORIG_MAIN' 
                    #for metric in 'PL_KS_distance'
                    do
                        sbatch scripts/plot_scatterplot.sh $metric $bleu_type $group power_law $fitting_method $dataset $size_param $adjust_measure
                    done
                    
                    # This is for TPL
                    for metric in  'E_TPL_lambda' 'E_TPL_KS_distance' 'E_TPL_beta' 'alpha_weighted' 'log_alpha_norm'
                    #for metric in 'E_TPL_KS_distance'
                    do
                        sbatch scripts/plot_scatterplot.sh $metric $bleu_type $group truncated_power_law $fitting_method $dataset $size_param $adjust_measure
                    done


                    # This is for EXP
                    for metric in 'EXP_lambda'
                    #for metric in
                    do
                                sbatch scripts/plot_scatterplot.sh $metric $bleu_type $group exponential $fitting_method $dataset $size_param $adjust_measure
                    done
                done
            done
        done
    done
done