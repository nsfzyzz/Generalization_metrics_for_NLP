'''
Plots correlation between BLEU scores and metrics.
'''
import argparse, pickle, json, os, re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from Simpson_metrics import METRIC_TYPES, METRIC_FILES
import pickle

def get_corr_df(metrics_df):
    '''
    Correlations for a single checkpoint
    '''
    correlations = []
    for metric, metric_type in METRIC_TYPES.items():
        # TODO: If doing phases, need to change this to calculate correlation within each phase
        # corr, _ = spearmanr(metrics_df['ood_bleu'], metrics_df[metric])
        corr, _ = spearmanr(metrics_df['id_bleu'], metrics_df[metric])
        if metric == 'rand_distance':
            corr = corr * -1.0
        correlations.append((metric, metric_type, corr))
    
    data = list(zip(*correlations))     # list of length 3: element 0 is metric names, element 1 is metric types, element 2 is correlations
    corr_df = pd.DataFrame(data={
        'metric': data[0],
        'type': data[1],
        'correlation': data[2]
    })
    return corr_df

def plot_correlation_single(checkpoint, metrics_df):
    '''
    Plots the correlations for a single checkpoint
    '''
    # TODO: update this to include depth and LR
    num_samples = re.search("sample_(\d+)", checkpoint).group(1)
    dropout = "no-dropout" if "dropout" in checkpoint else "normal"
    id = "WMT" if "WMT" in checkpoint else "IWSLT"
    
    fig, ax = plt.subplots(figsize=(3,6), dpi=150)
    fig.suptitle(f"num_samples: {num_samples}\ndropout: {dropout}")

    corr_df = get_corr_df(metrics_df)

    sns.barplot(
        data=corr_df,
        x='correlation',
        y='metric',
        hue='type',
        palette='deep',     # hue colors
        ax=ax,
        orient='h',
        order=corr_df.sort_values('correlation', ascending=False)['metric'],
        color='tab:blue',
    )
    ax.set_xlim([-1.0, 1.0])
    ax.legend(prop={'size': 6})
    plt.savefig(
        f"plots/ood_correlations/id-{id}_num_samples-{num_samples}_dropout-{dropout}",
        # f"plots/id_correlations/id-{id}_num_samples-{num_samples}_dropout-{dropout}",
        bbox_inches='tight'
    )

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
    
    for metric, _ in METRIC_TYPES.items():
        metric_vals = []

        if METRIC_FILES[metric] == 'results':
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
                        print(f"{FILE}\n\tepoch {epoch} missing {metric}")
            elif metric == 'exp_dist_exponent':
                d = results_EXP
                for epoch in epochs:
                    metric_vals.append(d[epoch]['details']['exponent'].mean())
            else:
                d = results_TPL
                for epoch in epochs:
                    if metric == 'ETPL_KS_distance':
                        metric_vals.append(d[epoch]['details']['D'].mean())
                    elif metric == 'TPL_alpha':
                        metric_vals.append(d[epoch]['details']['alpha'].mean())
                    elif metric in d[epoch]['details']:
                        metric_vals.append(d[epoch]['details'][metric].mean())
                    else:
                        metric_vals.append(np.nan)
                        print(f"{FILE}\n\tepoch {epoch} missing {metric}")
        
        elif METRIC_FILES[metric] == 'robust':
            margin_metrics = results_robust
            for epoch in epochs:
                if metric in margin_metrics[epoch]:
                    metric_vals.append(margin_metrics[epoch][metric])
                else:
                    # Fill in missing metrics with null (not all checkpoints have all metrics calculated)
                    metric_vals.append(np.nan)
                    print(f"{FILE}\n\tepoch {epoch} missing {metric}")
        
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
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--bleu_type", type=str, default='test', choices=['test', 'gap'])

    args = parser.parse_args()

    assert args.id
    ood = 'WMT' if args.id == 'IWSLT' else 'IWSLT'
    
    if args.checkpoint_dir:
        # Plot correlations for one checkpoint
        metrics_df = get_metrics_df(args.checkpoint_dir, args.bleu_type)
        plot_correlation_single(args.checkpoint_dir, metrics_df)
    
    else:
        # Plot correlations across all experiments
        from Simpson_experiments import EXPERIMENTS
        exps = EXPERIMENTS[f"{args.id}"] #+ EXPERIMENTS[ood]
        all_metrics = [get_metrics_df(exp, args.bleu_type) for exp in exps]
        corr_dfs = [get_corr_df(metric_df) for metric_df in all_metrics]
        all_corrs = pd.concat(corr_dfs)

        # Make plot
        fig, ax = plt.subplots(figsize=(6,10), dpi=150)
        fig.suptitle("Distribution of Rank Correlation\nbetween ID BLEU Score and Generalization Metric")
        
        pickle.dump(all_corrs, open(f'results/Simpson_correlation_{args.id}_{args.bleu_type}.pkl', "wb"))
        
        sns.boxplot(
            data=all_corrs,
            x='correlation',
            y='metric',
            hue='type',
            palette='pastel',     # hue colors
            ax=ax,
            orient='h',
            order=all_corrs.groupby("metric")['correlation'].median().sort_values(ascending=False).index,
        )
        ax.set_xlim([-1.0, 1.0])
        ax.legend(prop={'size': 6})
        plt.savefig(
            f"plots/id_correlations_{args.id}_{args.bleu_type}",
            # f"plots/ood_correlations/{args.id}",
            # f"plots/id_correlations/{args.id}",
            bbox_inches='tight'
        )