METRIC_FILES = {
    # robust
    'L2': 'robust',
    'L2_DIST': 'robust',
    'PARAM_NORM': 'robust',
    'FRO_DIST': 'robust',
    'LOG_SUM_OF_FRO': 'robust',
    'DIST_SPEC_INIT': 'robust',
    'LOG_PROD_OF_FRO': 'robust',
    'LOG_SUM_OF_SPEC': 'robust',
    'LOG_PROD_OF_SPEC': 'robust',
    'PATH_NORM': 'robust',
    'INVERSE_MARGIN': 'robust',
    'LOG_PROD_OF_SPEC_OVER_MARGIN': 'robust',
    'LOG_SUM_OF_SPEC_OVER_MARGIN': 'robust',
    'LOG_PROD_OF_FRO_OVER_MARGIN': 'robust',
    'LOG_SUM_OF_FRO_OVER_MARGIN': 'robust',
    'PATH_NORM_OVER_MARGIN': 'robust',
    'PACBAYES_INIT': 'robust',
    'PACBAYES_ORIG': 'robust',
    'PACBAYES_FLATNESS': 'robust',
    'PACBAYES_MAG_INIT': 'robust',
    'PACBAYES_MAG_ORIG': 'robust',
    'PACBAYES_MAG_FLATNESS': 'robust',
    'W_CKA': 'robust',
    # 'FRO_OVER_SPEC': 'robust',  # Repeated with Stable Rank
    'LOG_SPEC_INIT_MAIN': 'robust',
    'LOG_SPEC_ORIG_MAIN': 'robust',

    # ww
    'log_norm': 'ww',
    'log_spectral_norm': 'ww',
    'mp_softrank': 'ww',
    'stable_rank': 'ww',
    'PL_alpha': 'ww',
    'E_TPL_beta': 'ww',
    'E_TPL_lambda': 'ww',
    'EXP_lambda': 'ww',
    'PL_KS_distance': 'ww',
    'E_TPL_KS_distance': 'ww',
    'tail_mean_vec_entropy': 'ww',
    'bulk_mean_vec_entropy': 'ww',
    'entropy': 'ww',
    'rand_distance': 'ww',
    'alpha_weighted': 'ww',
    'log_alpha_norm': 'ww',
    #'logdet_tpl_per_layer': 'ww', # Testing this combined measure.
    #'exponent_adjusted': 'ww', # Testing this combined measure.
    
    # combined metrics calculated from existing ones
    #'logdet_tpl': 'combine'  # Testing this combined measure.
}