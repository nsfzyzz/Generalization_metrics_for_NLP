import re
import pickle
from scipy import stats
import numpy as np
import os

def change_dict_form(mypath):
        
    with open(mypath, 'rb') as f:
        dict0 = pickle.load(f)
    
    dict1 = {}
    for key in dict0[0].keys():
        dict1[key] = [dict0[epoch][key] for epoch in dict0.keys()]
        
    return dict1


def compute_correlation_single(a,b,correlation_type='pearsonr'):
    
    if correlation_type=='pearsonr':
        rho, pval = stats.pearsonr(a, b)
    elif correlation_type=='spearmanr':
        rho, pval = stats.spearmanr(a, b)
        
    return rho


def compute_rank_correlation(plot_result, different_metrics, exclude_metrics=[], correlation_type='pearsonr', single_features=[], bleu_score_type='test'):
    
    rank_correlations = {}
    if bleu_score_type=='test':
        bleu = [-x for x in plot_result['bleu_score']]
    elif bleu_score_type == 'gap':
        bleu = plot_result['bleu_score']
    else:
        raise ValueError('BLEU score type not implemented!')
        
    for key in different_metrics.keys():
        if key in exclude_metrics:
            continue
            
        rho = compute_correlation_single(bleu,different_metrics[key],correlation_type=correlation_type)
        rank_correlations[key] = rho
        
    rho = compute_correlation_single(bleu, plot_result['alpha'],correlation_type=correlation_type)
    rank_correlations['alpha'] = rho
    
    for key in single_features:
        rho = compute_correlation_single(bleu, plot_result[key],correlation_type=correlation_type)
        rank_correlations[key] = rho
        if key == 'rand_distance':
            rank_correlations[key] = -rho
        
    return rank_correlations


def create_plot_result(keys, single_features, exclude_metrics):
    
    plot_result = {key:[] for key in keys}
    plot_result.update({key:[] for key in ['exp_dist_exponent', 'lognormal_sigma']})
    plot_result.update({key:[] for key in single_features if key not in exclude_metrics})
    
    return plot_result


def aggregate_rank_correlations(rank_correlations, rank_correlations_min, rank_correlations_ave):
    
    if len(rank_correlations_min)==0:
        
        for key in rank_correlations.keys():
            rank_correlations_min[key] = rank_correlations[key]
            rank_correlations_ave[key] = rank_correlations[key]
        
        return
        
    for key in rank_correlations.keys():
        rank_correlations_min[key] = min(rank_correlations_min[key], rank_correlations[key])
        rank_correlations_ave[key] = rank_correlations_ave[key] + rank_correlations[key]
        
def average_rank_correlations(rank_correlations_ave):       
    
    for key in rank_correlations_ave:
        rank_correlations_ave[key] /= 18
        

def plot_rank_correlations(rank_correlations, ax):
    
    plot_keys = [k.upper() for k in rank_correlations.keys()]
    values = rank_correlations.values()

    y_pos = np.arange(len(plot_keys))

    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_keys)
    
    
def get_ESD(ckpt_folder, epoch, ESD_type=1, layer_id=105, metric_folder_name='metrics_expcutoff'):
    
    if ESD_type == 1:
        esd_suffix = f'esd'
    elif ESD_type in [2,3,4]:
        esd_suffix = f'esd{ESD_type}'
    elif ESD_type in [5,6]:
        esd_suffix = f'mpfit{ESD_type-4}'
    elif ESD_type in [7,8]:
        esd_suffix = f'randesd.{ESD_type-6}'

    ESD_result = os.path.join(ckpt_folder, metric_folder_name, f'epoch_{epoch}', f'ww.layer{layer_id}.{esd_suffix}.png')

    return ESD_result
