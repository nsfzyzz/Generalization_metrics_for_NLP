#!/bin/bash

# Two grids
datasets=("WMT14" "WMT14")
size_params=("depth" "width")

# For each grid, we calculate both the bleu scores and the generalization gap (train bleu - test bleu)
bleu_types=("id_bleu" "id_bleu_gap")

# For this experiment, we only calculate the rank correlations. The plots are generated from another bash file.
calculation_parameter="calculate"

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
                    for metric in 'PL_alpha' 'PL_KS_distance' 'mp_softrank' 'log_norm' 'log_spectral_norm' 'PARAM_NORM' 'FRO_DIST' 'DIST_SPEC_INIT' 'PATH_NORM' 'stable_rank' 'alpha_weighted' 'log_alpha_norm' 'INVERSE_MARGIN' 'LOG_PROD_OF_SPEC_OVER_MARGIN' 'LOG_SUM_OF_SPEC_OVER_MARGIN' 'LOG_PROD_OF_FRO_OVER_MARGIN' 'LOG_SUM_OF_FRO_OVER_MARGIN' 'PATH_NORM_OVER_MARGIN' 'PACBAYES_INIT' 'PACBAYES_ORIG' 'PACBAYES_FLATNESS' 'PACBAYES_MAG_INIT' 'PACBAYES_MAG_ORIG' 'PACBAYES_MAG_FLATNESS' 
                    do
                        sbatch scripts/hyperparameter_correlation.sh $metric $bleu_type $group power_law $fitting_method $dataset $calculation_parameter $size_param $adjust_measure
                    done
                    
                    # This is for TPL
                    for metric in  'E_TPL_lambda' 'E_TPL_KS_distance' 'E_TPL_beta' 'alpha_weighted' 'log_alpha_norm'
                    do
                        sbatch scripts/hyperparameter_correlation.sh $metric $bleu_type $group truncated_power_law $fitting_method $dataset $calculation_parameter $size_param $adjust_measure
                    done


                    # This is for EXP
                    for metric in 'EXP_lambda'
                    do
                        sbatch scripts/hyperparameter_correlation.sh $metric $bleu_type $group exponential $fitting_method $dataset $calculation_parameter $size_param $adjust_measure
                    done
                done
            done
        done
    done
done