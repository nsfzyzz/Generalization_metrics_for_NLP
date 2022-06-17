#!/bin/bash
for metric in 'PL_alpha' 'KS_distance' 'rand_distance' 'mp_softrank' 'log_norm' 'log_spectral_norm' 'L2' 'L2_DIST' 'PARAM_NORM' 'FRO_DIST' 'LOG_SUM_OF_FRO' 'DIST_SPEC_INIT' 'LOG_PROD_OF_FRO' 'LOG_SUM_OF_SPEC' 'LOG_PROD_OF_SPEC' 'PATH_NORM' 'stable_rank' 'tail_mean_vec_entropy' 'bulk_mean_vec_entropy' 'entropy' 'alpha_weighted' 'log_alpha_norm' 'INVERSE_MARGIN' 'LOG_PROD_OF_SPEC_OVER_MARGIN' 'LOG_SUM_OF_SPEC_OVER_MARGIN' 'LOG_PROD_OF_FRO_OVER_MARGIN' 'LOG_SUM_OF_FRO_OVER_MARGIN' 'PATH_NORM_OVER_MARGIN' 'PACBAYES_INIT' 'PACBAYES_ORIG' 'PACBAYES_FLATNESS' 'PACBAYES_MAG_INIT' 'PACBAYES_MAG_ORIG' 'PACBAYES_MAG_FLATNESS' 'W_CKA' 'FRO_OVER_SPEC' 'LOG_SPEC_INIT_MAIN' 'LOG_SPEC_ORIG_MAIN'
do
    for bleu_type in 'id_bleu' 'ood_bleu'
    do
        for group in 'sample' 'depth' 'lr'
        do
            sbatch scripts/plot_scatterplot.sh $metric $bleu_type $group
        done
    done
done
