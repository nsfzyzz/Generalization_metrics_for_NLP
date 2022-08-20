from experiments import EXPERIMENTS
#from experiments_single_depth import EXPERIMENTS
from metrics import METRIC_FILES
import argparse, pickle, os, json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpmath
import numpy as np
import pickle


def logdet_tpl_scalar(lam, beta):
    numer = mpmath.meijerg([[],[beta,beta]],[[0,-1+beta,-1+beta],[]],lam)
    denom = mpmath.expint(beta,lam)
    return float(numer / denom)

logdet_tpl = np.vectorize(logdet_tpl_scalar)

def adjust_measure(metric, val, dataset_size):
    
    if metric.startswith('LOG_'):
        return 2*val + np.log(dataset_size)
        #return 0.5 * (value - np.log(m))
    elif 'CKA' in metric or 'TRUE_MARGIN' in metric:
        return val
    else:
        #print(val)
        #print(dataset_size)
        return (val**2)*dataset_size
        #return np.sqrt(value / m)

# TODO: Move into a utils file
def get_metric_bleu_df(experiment, distribution, adjust_measures_back):
    '''
    Constructs a DataFrame of length num_epochs.
    The columns are [epoch, id_bleu, ood_bleu, metric1, metric2, ...]
    '''
    print(experiment)

    ### Get metrics ###
    metrics = {}    # Key: metric name, Value: list of metric values (length num_epochs)

    for metric, metric_file in METRIC_FILES.items():
        metric_vals = []
        
        #print("Retrieving the metric:")
        #print(metric)
        # Special cases: PL vs TPL alpha
        if metric in ['PL_alpha', 'TPL_alpha']:
            if metric == 'PL_alpha':
                FILE = os.path.join(experiment, "results_original_alpha.pkl")
            elif metric == 'TPL_alpha':
                FILE = os.path.join(experiment, "results.pkl")
            with open(FILE, 'rb') as file:
                d = pickle.load(file)
            epochs = d.keys()
            for epoch in epochs:
                metric_vals.append(d[epoch]['details']['alpha'].mean())     # averaging over layers
                
        elif metric == 'exponent':
            FILE = os.path.join(experiment, "results.pkl")
            with open(FILE, 'rb') as file:
                d = pickle.load(file)
            epochs = d.keys()
            for epoch in epochs:
                metric_vals.append(d[epoch]['details']['exponent'].mean())     # averaging over layers
                
        elif metric == 'exponent_adjusted':
            FILE = os.path.join(experiment, "results.pkl")
            with open(FILE, 'rb') as file:
                d = pickle.load(file)
            epochs = d.keys()
            for epoch in epochs:
                exp_adjusted = [exp_layer*xmin_layer for exp_layer, xmin_layer in zip(d[epoch]['details']['exponent'], d[epoch]['details']['xmin'])]     # adjust the exponent by xmin
                exp_adjusted = np.array(exp_adjusted).mean()  # averaging over layers
                metric_vals.append(exp_adjusted)     
        
        elif metric_file == 'ww':
            # Get from results.pkl
            if distribution == "PL":
                FILE = os.path.join(experiment, "results_original_alpha.pkl")
            elif distribution == "TPL":
                FILE = os.path.join(experiment, "results.pkl")
            else:
                raise ValueError('Unknown distribution.')
            with open(FILE, 'rb') as file:
                d = pickle.load(file)
            epochs = d.keys()
            for epoch in epochs:
                # Special case for KS_distance
                if metric == 'KS_distance':
                    metric_vals.append(d[epoch]['details']['D'].mean())     # averaging over layers
                elif metric == 'logdet_tpl_per_layer':
                    betas = d[epoch]['details']['alpha']
                    lambdas = d[epoch]['details']['exponent']
                    metric_vals.append(logdet_tpl(lambdas, betas).mean())
                elif metric in d[epoch]['details']:
                    metric_vals.append(d[epoch]['details'][metric].mean())     # averaging over layers
                else:
                    print(f"{FILE} missing {metric}")
                    metric_vals.append(np.nan)
                    
        elif metric_file == 'robust':
            # Get from robust_measures.pkl
            FILE = os.path.join(experiment, "robust_measures.pkl")
            with open(FILE, 'rb') as file:
                d = pickle.load(file)
            epochs = d.keys()
            for epoch in epochs:
                if metric in d[epoch]:
                    _val = d[epoch][metric]
                    if adjust_measures_back:
                        # Reverse the effect of dataset_size
                        dataset_size = int(re.search("sample(\d+)", experiment).group(1))
                        _val_adjusted = adjust_measure(metric, _val, dataset_size)
                        metric_vals.append(_val_adjusted)
                    else:
                        metric_vals.append(_val)
                else:
                    print(f"{FILE} missing {metric}")
                    metric_vals.append(np.nan)
        
        elif metric_file == 'combine':
            # These are the metrics combined by existing metrics
            
            if metric == 'logdet_tpl':
                assert 'exponent' in metrics.keys()
                assert 'TPL_alpha' in metrics.keys()
                lambdas = metrics['exponent']
                betas = metrics['TPL_alpha']
                metric_vals = logdet_tpl(lambdas, betas)
            else:
                raise ValueError('Combined metric not found.')
        
        metrics[metric] = metric_vals
    
    ### Get BLEU scores ###
    id_bleu_scores, ood_bleu_scores, id_bleu_gaps, id_bleu_train_scores, id_loss_gaps, id_train_losses, id_val_losses = [], [], [], [], [], [], []
    EPOCH = 1   # Epochs are numbered 1-20
    FILE = os.path.join(experiment, "bleu_loss.jsonl")
    with open(FILE, "rb") as file:
        for line in file:
            d = json.loads(line)
            id_bleu_scores.append(d[f'epoch{EPOCH}_id_bleu_score'] * 100)
            ood_bleu_scores.append(d[f'epoch{EPOCH}_ood_bleu_score'] * 100)
            id_bleu_train_scores.append(d[f'epoch{EPOCH}_id_train_bleu_score'] * 100)
            id_bleu_gaps.append((d[f'epoch{EPOCH}_id_train_bleu_score'] - d[f'epoch{EPOCH}_id_bleu_score'])* 100)
            id_loss_gaps.append(d[f'epoch{EPOCH}_id_val_loss'] - d[f'epoch{EPOCH}_id_train_loss'])
            id_train_losses.append(d[f'epoch{EPOCH}_id_train_loss'])
            id_val_losses.append(d[f'epoch{EPOCH}_id_val_loss'])
            
            EPOCH += 1
    
    ### Construct the DataFrame ###
    data = {
        'epoch': epochs, 'id_bleu': id_bleu_scores, 'ood_bleu': ood_bleu_scores, 
        'id_bleu_gap': id_bleu_gaps, 'id_bleu_train': id_bleu_train_scores, 'id_loss_gap': id_loss_gaps,
        'id_loss_train': id_train_losses, 'id_loss_val': id_val_losses
    }
    data.update(metrics)
    #num_epochs = len(data['epoch'])
    #for key in data.keys():
    #    if len(data[key])<num_epochs:
    #        data[key] = [0] + data[key]
    
    try:
        df = pd.DataFrame(data=data)
    except ValueError:
        print('The dimension does not match! The experiment is')
        print(experiment)
        for key in data.keys():
            print(key + " dimension is "+str(len(data[key])))
        
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="")
    parser.add_argument("--bleu_type", type=str)
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--distribution", type=str, default="PL")
    parser.add_argument("--adjust_measures_back", dest='adjust_measures_back', action='store_true', help='adjust the measure back using the dataset size (default: off)')

    args = parser.parse_args()
    assert args.metric in METRIC_FILES.keys()
    assert args.bleu_type in ["id_bleu", "ood_bleu", "id_bleu_gap", "id_bleu_train", "id_loss_gap", "id_loss_train", "id_loss_val"]
    assert args.group in ["sample", "depth", "lr"]
    assert args.distribution in ["PL", "TPL"]

    # Construct a DataFrame of length num_experiments
    # The columns are [id_bleu, ood_bleu, metric, sample, depth, lr, dropout]
    records = []
    for experiment in EXPERIMENTS:
        metric_bleu_df = get_metric_bleu_df(experiment, args.distribution, args.adjust_measures_back)
        # Get the last three epochs' BLEU/metric
        average_length = 3
        record = {
            'id_bleu': sum([metric_bleu_df.iloc[-x]['id_bleu'] for x in range(1,1+average_length)])/average_length,
            'id_bleu_train': sum([metric_bleu_df.iloc[-x]['id_bleu_train'] for x in range(1,1+average_length)])/average_length,
            'id_bleu_gap': sum([metric_bleu_df.iloc[-x]['id_bleu_gap'] for x in range(1,1+average_length)])/average_length,
            'id_loss_gap': sum([metric_bleu_df.iloc[-x]['id_loss_gap'] for x in range(1,1+average_length)])/average_length,
            'id_loss_train': sum([metric_bleu_df.iloc[-x]['id_loss_train'] for x in range(1,1+average_length)])/average_length,
            'id_loss_val': sum([metric_bleu_df.iloc[-x]['id_loss_val'] for x in range(1,1+average_length)])/average_length,
            'ood_bleu': sum([metric_bleu_df.iloc[-x]['ood_bleu'] for x in range(1,1+average_length)])/average_length,
            f'{args.metric}': sum([metric_bleu_df.iloc[-x][f'{args.metric}'] for x in range(1,1+average_length)])/average_length,
            'sample': re.search("sample(\d+)", experiment).group(1),
            'depth': re.search("depth(\d+)", experiment).group(1),
            'lr': re.search("lr([\d.]+)", experiment).group(1),
            'dropout': re.search("dropout([\d.]+)", experiment).group(1),
        }
        records.append(record)
    
    df = pd.DataFrame.from_records(records)
    
    plot_metric_name = args.metric.lower()
    if plot_metric_name == 'exponent':
        plot_metric_name = 'E_TPL_lambda'
    elif plot_metric_name == 'tpl_alpha':
        plot_metric_name = 'E_TPL_beta'
    elif plot_metric_name == 'pl_alpha':
        plot_metric_name = 'PL_alpha'
    elif plot_metric_name == 'exponent_adjusted':
        plot_metric_name = 'E_TPL_lambda_adjusted'
        
    plot_bleu_type_name = args.bleu_type
    if plot_bleu_type_name == 'id_bleu':
        plot_bleu_type_name = 'BLEU score'
        
    plot_group_name = args.group
    if plot_group_name == 'lr':
        plot_group_name = 'Learning rate'
    if plot_group_name == 'sample':
        plot_group_name = 'Num samples'
    
    ### Compute spearman's rank correlations
    SAVE_DIR_CORR = f"results/{args.distribution}/{plot_metric_name}"
    if not os.path.exists(SAVE_DIR_CORR):
        os.makedirs(SAVE_DIR_CORR)

    rank_correlation_result = {'sample':[], 'depth':[], 'lr':[]}
    for g0, g1, g2 in [('sample', 'depth', 'lr'), ('depth', 'lr', 'sample'), ('lr', 'sample', 'depth')]:
        for g1_value in df[g1].unique():
            for g2_value in df[g2].unique():
                corr = df.loc[ (df[g1] == g1_value) & (df[g2] == g2_value)][[args.bleu_type, args.metric]].corr(method='spearman').values[0][1]
                rank_correlation_result[g0].append(corr)

    pickle.dump(rank_correlation_result, open(os.path.join(SAVE_DIR_CORR, f'corr_{args.bleu_type}.pkl'), 'wb'))

    ### Make scatterplots ###
    SAVE_DIR = f"plots/{args.distribution}/{plot_metric_name}"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Regular scatterplot
    fig, ax = plt.subplots(figsize=(9,9))
    lm = sns.lmplot(
        data=df,
        x=f'{args.metric}',
        y=f'{args.bleu_type}',
        hue=f'{args.group}',
        fit_reg=False,
        legend=False,
    )
    ax = lm.axes[0, 0]
    ax.set_xlabel(plot_metric_name, fontsize=18)
    ax.set_ylabel(plot_bleu_type_name, fontsize=18)
    ax.set_title(f"{plot_metric_name} vs. {plot_bleu_type_name}", fontsize=18)
    legend = plt.legend(title=plot_group_name, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)#, labels=['Hell Yeh', 'Nah Bruh'])
    plt.setp(legend.get_title(),fontsize=14)
    plt.savefig(
        os.path.join(SAVE_DIR, f"{args.bleu_type}_{plot_metric_name}_{args.group}"),
        bbox_inches='tight',
        dpi=150,
    )
    xmin,xmax,ymin,ymax = plt.axis()    # save for making best within group plot

    # Simpson's scatterplot
    fig, ax = plt.subplots(figsize=(9,9))
    lm = sns.lmplot(
        data=df,
        x=f'{args.metric}',
        y=f'{args.bleu_type}',
        hue=f'{args.group}',
        fit_reg=True,
        ci=None,
        legend=False,
    )
    ax = lm.axes[0, 0]
    sns.regplot(
        data=df,
        x=f'{args.metric}',
        y=f'{args.bleu_type}',
        scatter=False,
        fit_reg=True,
        ci=None,
        color='gray',
    )    
    ax.set_xlabel(plot_metric_name, fontsize=18)
    ax.set_ylabel(plot_bleu_type_name, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_title(f"{plot_metric_name} vs. {plot_bleu_type_name}", fontsize=18)
    legend = plt.legend(title=plot_group_name, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
    plt.setp(legend.get_title(),fontsize=14)
    plt.savefig(
        os.path.join(SAVE_DIR, f"{args.bleu_type}_{plot_metric_name}_{args.group}_simpson"),
        bbox_inches='tight',
        dpi=150,
    )

    # Only best performing in each group
    fig, ax = plt.subplots(figsize=(9,9))
    lm = sns.lmplot(
        data=df.sort_values(by=f'{args.bleu_type}', ascending=False).groupby(f'{args.group}', as_index=False).first(),
        x=f'{args.metric}',
        y=f'{args.bleu_type}',
        hue=f'{args.group}',
        fit_reg=False,
        legend=False,
    )
    ax = lm.axes[0, 0]
    ax.set_xlabel(plot_metric_name, fontsize=18)
    ax.set_ylabel(plot_bleu_type_name, fontsize=18)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_title(f"{plot_metric_name} vs. {plot_bleu_type_name}\nbest performing model in each group", fontsize=18)
    legend = plt.legend(title=plot_group_name, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
    plt.setp(legend.get_title(),fontsize=14)
    plt.savefig(
        os.path.join(SAVE_DIR, f"{args.bleu_type}_{plot_metric_name}_{args.group}_best"),
        bbox_inches='tight',
        dpi=150,
    )
