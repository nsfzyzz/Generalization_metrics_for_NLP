'''
This file calculates time-wise correlation between BLEU scores and metrics.
'''
import argparse, pickle, json, os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from metrics import METRIC_FILES
import pickle


def get_corr_df(metrics_df):
    '''
    Correlations for a single checkpoint
    '''
    correlations = []
    for metric, _ in METRIC_FILES.items():
        # TODO: If doing phases, need to change this to calculate correlation within each phase
        # corr, _ = spearmanr(metrics_df['ood_bleu'], metrics_df[metric])
        corr, _ = spearmanr(metrics_df['id_bleu'], metrics_df[metric])
        if metric == 'rand_distance':
            corr = corr * -1.0
        correlations.append((metric, corr))
    
    data = list(zip(*correlations))     # list of length 3: element 0 is metric names, element 1 is metric types, element 2 is correlations
    corr_df = pd.DataFrame(data={
        'metric': data[0],
        'correlation': data[1]
    })
    return corr_df


def get_metrics_df(checkpoint, bleu_type = 'test'):
    '''
    Create a dataframe of metrics and BLEU scores for a given checkpoint directory
    '''
    print(checkpoint)
    # Get ww metrics
    ww_metrics = {}     # Key: metric, Value: list of values for that metric
    
    # Epochs are numbered 1-20
    EPOCHS = 20
    epochs = range(1, EPOCHS+1)
    
    # Load results
    FILE_PL = os.path.join(checkpoint, f"results_original_alpha.pkl")
    with open(FILE_PL, "rb") as file:
        results_PL = pickle.load(file)
    FILE_TPL = os.path.join(checkpoint, f"results.pkl")
    with open(FILE_TPL, "rb") as file:
        results_TPL = pickle.load(file)
    FILE_EXP = os.path.join(checkpoint, f"results_exponential.pkl")
    with open(FILE_EXP, "rb") as file:
        results_EXP = pickle.load(file)
    FILE_ROBUST = os.path.join(checkpoint, f"robust_measures.pkl")
    with open(FILE_ROBUST, "rb") as file:
        results_robust = pickle.load(file)
    
    for metric, _ in METRIC_FILES.items():
        metric_vals = []

        if METRIC_FILES[metric] == 'ww':
            if metric in ['PL_alpha', 'rand_distance', 'mp_softrank', 'PL_KS_distance', 'alpha_weighted', 'log_alpha_norm', 'stable_rank']:
                for epoch in epochs:
                    results_metrics = results_PL[epoch]
                    if  metric == 'PL_alpha':
                    #if 'alpha' in results_metrics['details']:
                        metric_vals.append(results_metrics['details']['alpha'].mean())             # averaging over layers
                    elif metric == 'PL_KS_distance':
                        metric_vals.append(results_metrics['details']['D'].mean())
                    elif metric in d[epoch]['details']:
                        metric_vals.append(results_metrics['details'][metric].mean())
                    else:
                        # Fill in missing metrics with null (not all checkpoints have all metrics calculated)
                        metric_vals.append(np.nan)
                        print(f"{FILE_PL}\n\tepoch {epoch} missing {metric}")
            elif metric == 'EXP_lambda':
                d = results_EXP
                for epoch in epochs:
                    metric_vals.append(d[epoch]['details']['exponent'].mean())
            else:
                d = results_TPL
                for epoch in epochs:
                    if metric == 'E_TPL_KS_distance':
                        metric_vals.append(d[epoch]['details']['D'].mean())
                    elif metric == 'E_TPL_beta':
                        metric_vals.append(d[epoch]['details']['alpha'].mean())
                    elif metric == 'E_TPL_lambda':
                        metric_vals.append(d[epoch]['details']['exponent'].mean())
                    elif metric in d[epoch]['details']:
                        metric_vals.append(d[epoch]['details'][metric].mean())
                    else:
                        metric_vals.append(np.nan)
                        print(f"{FILE_TPL}\n\tepoch {epoch} missing {metric}")
        
        elif METRIC_FILES[metric] == 'robust':
            margin_metrics = results_robust
            for epoch in epochs:
                if metric in margin_metrics[epoch]:
                    metric_vals.append(margin_metrics[epoch][metric])
                else:
                    # Fill in missing metrics with null (not all checkpoints have all metrics calculated)
                    metric_vals.append(np.nan)
                    print(f"{FILE_ROBUST}\n\tepoch {epoch} missing {metric}")
        
        else:
            print(f"{metric} not found")
        
        ww_metrics[metric] = metric_vals
    
    # Get BLEU scores
    id_bleu_scores, ood_bleu_scores = [], []
    FILE = os.path.join(checkpoint, "bleu_loss.jsonl")
    
    EPOCH = 1   # Epochs are numbered 1-20
    with (open(FILE, "rb")) as file:
        for line in file:
            d = json.loads(line)
            # Multiply BLEU by -1 because we are computing correlations between BLEU
            # and generalization metrics for which lower values are better
            if bleu_type == 'test':
                id_bleu_scores.append(d[f'epoch{EPOCH}_id_bleu_score'] * 100 * -1.0)
                ood_bleu_scores.append(d[f'epoch{EPOCH}_ood_bleu_score'] * 100 * -1.0)
            elif bleu_type == 'gap':
                #TODO: Results for OOD generalization gap
                id_bleu_scores.append((d[f'epoch{EPOCH}_id_train_bleu_score'] - d[f'epoch{EPOCH}_id_bleu_score'])* 100)
                ood_bleu_scores.append(d[f'epoch{EPOCH}_ood_bleu_score'] * 100 * -1.0)
            else:
                raise ValueError('Bleu type not implemented.')
            EPOCH += 1
    ###
        
    assert len(ww_metrics['log_spectral_norm']) == len(id_bleu_scores) == len(ood_bleu_scores)

    # Create a dataframe
    data={'epoch': list(range(1, EPOCHS+1)), 'id_bleu': id_bleu_scores, 'ood_bleu': ood_bleu_scores}
    data.update(ww_metrics)
    df = pd.DataFrame(data=data)
    ###
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="WMT")
    parser.add_argument("--bleu_type", type=str, default='test', choices=['test', 'gap'])
    #TODO: update the WW results using WeightWatcher 0.5.6
    #parser.add_argument("--reproduce", action='store_true')

    args = parser.parse_args()
    ood = 'WMT' if args.id == 'IWSLT' else 'IWSLT'

    # Plot correlations across all experiments
    from experiments_time_wise import EXPERIMENTS
    exps = EXPERIMENTS[f"{args.id}"] #+ EXPERIMENTS[ood]
    all_metrics = [get_metrics_df(exp, args.bleu_type) for exp in exps]
    corr_dfs = [get_corr_df(metric_df) for metric_df in all_metrics]
    all_corrs = pd.concat(corr_dfs)
    
    rank_correlations_aggregated = {}

    # Converting all results into an aggregated array
    for key, val in zip(all_corrs['metric'].values, all_corrs['correlation'].values):
        if key not in rank_correlations_aggregated:
            rank_correlations_aggregated[key] = [val]
        else:
            rank_correlations_aggregated[key].append(val)
    
    # Remove nan's which are failed measurements
    for key in rank_correlations_aggregated.keys():
        rank_correlations_aggregated[key] = [x for x in rank_correlations_aggregated[key] if not np.isnan(x)]
        
    with open(f'results/plot_results_{args.bleu_type}_Simpson_{args.id}.pkl', 'wb') as f:
        pickle.dump(rank_correlations_aggregated, f)
    
    #pickle.dump(all_corrs, open(f'results/Simpson_correlation_{args.id}_{args.bleu_type}.pkl', "wb"))