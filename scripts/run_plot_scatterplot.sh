#!/bin/bash

# WMT data
datasets=("WMT14")
size_params=("depth")

# For each data, we plot the scatter plots for both the bleu scores and the generalization gap (train bleu - test bleu)
bleu_types=("id_bleu" "id_bleu_gap")

# We do plots instead of calculating correlations
calculation_parameter="plot"

# Generate scatter plots for metrics that are not normalized by the number of training samples
for adjust_measure in '--adjust_measures_back'
do
    for i in ${!datasets[@]}; do
        dataset=${datasets[$i]}
        size_param=${size_params[$i]}

        # We test two fitting algorithms, namely linear regression and orthogonal distance regression
        for fitting_method in 'ODR' 'LR'
        do
        
            for bleu_type in ${bleu_types[@]}
            do
                for group in 'sample' 'depth' 'lr'
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