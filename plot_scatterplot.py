#from experiments import EXPERIMENTS
from experiments_single_depth import EXPERIMENTS
from metrics import METRIC_FILES
import argparse, pickle, os, json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    EPOCHS = 20
    epochs = list(range(1, EPOCHS+1))

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
            for epoch in epochs:
                metric_vals.append(d[epoch]['details']['alpha'].mean())     # averaging over layers
                
        elif metric == 'exponent':
            FILE = os.path.join(experiment, "results.pkl")
            with open(FILE, 'rb') as file:
                d = pickle.load(file)
            for epoch in epochs:
                metric_vals.append(d[epoch]['details']['exponent'].mean())     # averaging over layers
        
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
            for epoch in epochs:
                # Special case for KS_distance
                if metric == 'KS_distance':
                    metric_vals.append(d[epoch]['details']['D'].mean())     # averaging over layers
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
        
        metrics[metric] = metric_vals
    
    ### Get BLEU scores ###
    id_bleu_scores, ood_bleu_scores, id_bleu_gaps, id_bleu_train_scores, id_loss_gaps = [], [], [], [], []
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
            
            EPOCH += 1
    
    ### Construct the DataFrame ###
    data = {
        'epoch': epochs, 'id_bleu': id_bleu_scores, 'ood_bleu': ood_bleu_scores, 
        'id_bleu_gap': id_bleu_gaps, 'id_bleu_train': id_bleu_train_scores, 'id_loss_gap': id_loss_gaps,
    }
    data.update(metrics)
    df = pd.DataFrame(data=data)
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
    assert args.bleu_type in ["id_bleu", "ood_bleu", "id_bleu_gap", "id_bleu_train", "id_loss_gap"]
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
            'ood_bleu': sum([metric_bleu_df.iloc[-x]['ood_bleu'] for x in range(1,1+average_length)])/average_length,
            f'{args.metric}': sum([metric_bleu_df.iloc[-x][f'{args.metric}'] for x in range(1,1+average_length)])/average_length,
            'sample': re.search("sample(\d+)", experiment).group(1),
            'depth': re.search("depth(\d+)", experiment).group(1),
            'lr': re.search("lr([\d.]+)", experiment).group(1),
            'dropout': re.search("dropout([\d.]+)", experiment).group(1),
        }
        records.append(record)
    
    df = pd.DataFrame.from_records(records)
    
    plot_metric_name = args.metric
    if plot_metric_name == 'exponent':
        plot_metric_name = 'E_TPL_lambda'
    elif plot_metric_name == 'TPL_alpha':
        plot_metric_name = 'E_TPL_beta'
    
    ### Make scatterplots ###
    SAVE_DIR = f"plots/{args.distribution}/{plot_metric_name}"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Regular scatterplot
    fig, ax = plt.subplots(figsize=(9,9))
    sns.lmplot(
        data=df,
        x=f'{args.metric}',
        y=f'{args.bleu_type}',
        hue=f'{args.group}',
        fit_reg=False,
    )
    plt.xlabel(plot_metric_name)
    plt.title(f"{plot_metric_name} vs. {args.bleu_type}")
    plt.savefig(
        os.path.join(SAVE_DIR, f"{args.bleu_type}_{plot_metric_name}_{args.group}"),
        bbox_inches='tight',
        dpi=150,
    )
    xmin,xmax,ymin,ymax = plt.axis()    # save for making best within group plot

    # Simpson's scatterplot
    sns.lmplot(
        data=df,
        x=f'{args.metric}',
        y=f'{args.bleu_type}',
        hue=f'{args.group}',
        fit_reg=True,
        ci=None,
    )
    sns.regplot(
        data=df,
        x=f'{args.metric}',
        y=f'{args.bleu_type}',
        scatter=False,
        fit_reg=True,
        ci=None,
        color='gray',
    )
    plt.xlabel(plot_metric_name)
    plt.title(f"{plot_metric_name} vs. {args.bleu_type}")
    plt.savefig(
        os.path.join(SAVE_DIR, f"{args.bleu_type}_{plot_metric_name}_{args.group}_simpson"),
        bbox_inches='tight',
        dpi=150,
    )

    # Only best performing in each group
    fig, ax = plt.subplots(figsize=(9,9))
    sns.lmplot(
        data=df.sort_values(by=f'{args.bleu_type}', ascending=False).groupby(f'{args.group}', as_index=False).first(),
        x=f'{args.metric}',
        y=f'{args.bleu_type}',
        hue=f'{args.group}',
        fit_reg=False,
    )
    plt.xlabel(plot_metric_name)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.title(f"{plot_metric_name} vs. {args.bleu_type}\nbest performing model in each group")
    plt.savefig(
        os.path.join(SAVE_DIR, f"{args.bleu_type}_{plot_metric_name}_{args.group}_best"),
        bbox_inches='tight',
        dpi=150,
    )


