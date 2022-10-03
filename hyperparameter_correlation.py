from experiments_hyperparameters import EXPERIMENTS
from metrics import METRIC_FILES
import argparse, pickle, os, json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpmath
import numpy as np
import pickle
from scipy import odr 

def f(B, x):
    return B[0]*x + B[1]
        
def plot_odr(x,y,ax,color=''):
    
    data = odr.Data(x, y)
    linear = odr.Model(f)
    
    # Using linear regression to find the starting point
    lr_reg = odr.ODR(data, linear, beta0=[1., 2.])
    lr_reg.set_job(fit_type=2)
    lr_out = lr_reg.run()
    
    ordinal_distance_reg = odr.ODR(data, linear, beta0=lr_out.beta)
    ordinal_distance_reg.set_job(fit_type=0)
    out = ordinal_distance_reg.run()
    
    xx = np.linspace(min(x),max(x),100)
    #out.pprint()
    yy = out.beta[0]*xx + out.beta[1]
    # delete large y values
    valid_indices = (yy<=max(y)) & (yy>=min(y))
    yy = yy[valid_indices]
    xx = xx[valid_indices]
    if color:
        ax.plot(xx,yy,color=color,linewidth=2)
    else:
        ax.plot(xx,yy,linewidth=2)
        

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
def get_metric_bleu_df(experiment, distribution, adjust_measures_back, metric):
    '''
    Constructs a DataFrame of length num_epochs.
    The columns are [epoch, id_bleu, ood_bleu, metric1, metric2, ...]
    '''
    print(experiment)

    ### Get metrics ###
    metric_vals = []
    metric_file = METRIC_FILES[metric]

    # Special cases: PL vs TPL alpha
    if metric in ['PL_alpha', 'E_TPL_beta']:
        if metric == 'PL_alpha':
            FILE = os.path.join(experiment, "results_original_alpha.pkl")
        elif metric == 'E_TPL_beta':
            FILE = os.path.join(experiment, "results.pkl")
        with open(FILE, 'rb') as file:
            d = pickle.load(file)
        epochs = d.keys()
        for epoch in epochs:
            metric_vals.append(d[epoch]['details']['alpha'].mean())     # averaging over layers

    elif metric == 'E_TPL_lambda':
        FILE = os.path.join(experiment, "results.pkl")
        with open(FILE, 'rb') as file:
            d = pickle.load(file)
        epochs = d.keys()
        for epoch in epochs:
            metric_vals.append(d[epoch]['details']['exponent'].mean())     # averaging over layers

    elif metric == 'EXP_lambda':
        FILE = os.path.join(experiment, "results_exponential.pkl")
        with open(FILE, 'rb') as file:
            d = pickle.load(file)
        epochs = d.keys()
        for epoch in epochs:
            metric_vals.append(d[epoch]['details']['exponent'].mean())     # averaging over layers  

    elif metric_file == 'ww':
        # Get from results.pkl
        if distribution == "power_law":
            FILE = os.path.join(experiment, "results_original_alpha.pkl")
        elif distribution == "truncated_power_law":
            FILE = os.path.join(experiment, "results.pkl")
        else:
            raise ValueError('Unknown distribution.')
        with open(FILE, 'rb') as file:
            d = pickle.load(file)
        epochs = d.keys()
        for epoch in epochs:
            # Special case for KS_distance
            if metric in ['PL_KS_distance', 'E_TPL_KS_distance']:
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
    
    metrics = {}    # Key: metric name, Value: list of metric values (length num_epochs)
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
    parser.add_argument("--bleu_type", type=str, choices=["id_bleu", "ood_bleu", "id_bleu_gap", "id_bleu_train", "id_loss_gap", "id_loss_train", "id_loss_val"])
    parser.add_argument("--dataset", type=str, choices=["WMT14", "IWSLT"], default="WMT14")
    parser.add_argument("--group", type=str, default="sample", choices=["sample", "depth", "lr", "width"])
    parser.add_argument("--fitting_method", type=str, default="LR", choices=["LR", "ODR"])
    parser.add_argument("--distribution", type=str, default="power_law", choices=["power_law", "truncated_power_law", "exponential"])
    parser.add_argument("--model_size_param", type=str, default="depth", choices=["width", "depth"])
    parser.add_argument("--adjust_measures_back", dest='adjust_measures_back', action='store_true', help='adjust the measure back using the dataset size (default: off)')
    parser.add_argument("--calculate_or_plot", dest='calculate_or_plot', type=str, choices=["calculate", "plot", "both"])

    args = parser.parse_args()
    assert args.metric in METRIC_FILES.keys()

    # Construct a DataFrame of length num_experiments
    # The columns are [id_bleu, ood_bleu, metric, sample, depth, lr, dropout]
    records = []
    for experiment in EXPERIMENTS[f"{args.dataset}_{args.model_size_param}"]:
        metric_bleu_df = get_metric_bleu_df(experiment, args.distribution, args.adjust_measures_back, args.metric)
        # Get the last three epochs' BLEU/metric
        average_length = 6
        
        record = {
            'id_bleu': sum([metric_bleu_df.iloc[-x]['id_bleu'] for x in range(1,1+average_length)])/average_length,
            'id_bleu_train': sum([metric_bleu_df.iloc[-x]['id_bleu_train'] for x in range(1,1+average_length)])/average_length,
            'id_bleu_gap': sum([metric_bleu_df.iloc[-x]['id_bleu_gap'] for x in range(1,1+average_length)])/average_length,
            'id_loss_gap': sum([metric_bleu_df.iloc[-x]['id_loss_gap'] for x in range(1,1+average_length)])/average_length,
            'id_loss_train': sum([metric_bleu_df.iloc[-x]['id_loss_train'] for x in range(1,1+average_length)])/average_length,
            'id_loss_val': sum([metric_bleu_df.iloc[-x]['id_loss_val'] for x in range(1,1+average_length)])/average_length,
            'ood_bleu': sum([metric_bleu_df.iloc[-x]['ood_bleu'] for x in range(1,1+average_length)])/average_length,
            f'{args.metric}': sum([metric_bleu_df.iloc[-x][f'{args.metric}'] for x in range(1,1+average_length)])/average_length,
            'sample': int(re.search("sample(\d+)", experiment).group(1)),
            'depth': int(re.search("depth(\d+)", experiment).group(1)),
            'width': int(re.search("width(\d+)", experiment).group(1)),
            'lr': float(re.search("lr([\d.]+)", experiment).group(1)),
            'dropout': float(re.search("dropout([\d.]+)", experiment).group(1)),
        }
        records.append(record)
    
    df = pd.DataFrame.from_records(records)
    
    plot_metric_name = args.metric.lower()
    plot_bleu_type_name = args.bleu_type
    if plot_bleu_type_name == 'id_bleu':
        plot_bleu_type_name = 'BLEU score'
        
    plot_group_name = args.group
    if plot_group_name == 'lr':
        plot_group_name = 'Learning rate'
    if plot_group_name == 'sample':
        plot_group_name = 'Num samples'
    
    if args.calculate_or_plot in ["calculate", "both"]:
            
        ### Compute spearman's rank correlations ###
        if args.group == 'sample':
            SAVE_DIR_CORR = f"results/{args.dataset}_Simpson/{args.distribution}/{plot_metric_name}"
            if not os.path.exists(SAVE_DIR_CORR):
                os.makedirs(SAVE_DIR_CORR)
            
            if args.model_size_param == 'depth':
                rank_correlation_result = {'sample':[], 'depth':[], 'lr':[]}
                three_parameters_grids = [('sample', 'depth', 'lr'), ('depth', 'lr', 'sample'), ('lr', 'sample', 'depth')]
            elif args.model_size_param == 'width':
                rank_correlation_result = {'sample':[], 'width':[], 'lr':[]}
                three_parameters_grids = [('sample', 'width', 'lr'), ('width', 'lr', 'sample'), ('lr', 'sample', 'width')]
            
            for g0, g1, g2 in three_parameters_grids:
                for g1_value in df[g1].unique():
                    for g2_value in df[g2].unique():
                        one_slice = df.loc[ (df[g1] == g1_value) & (df[g2] == g2_value)][[args.bleu_type, args.metric]]
                        corr = one_slice.corr(method='spearman').values[0][1]
                        rank_correlation_result[g0].append(corr)
            
            if args.adjust_measures_back:
                adjust_measures_suffix = 'not_normalized_by_samples'
            else:
                adjust_measures_suffix = 'normalized_by_samples'
                
            pickle.dump(rank_correlation_result, open(os.path.join(SAVE_DIR_CORR, f'corr_{args.bleu_type}_{args.dataset}_size_param_{args.model_size_param}_{adjust_measures_suffix}.pkl'), 'wb'))
                
    if args.calculate_or_plot in ["plot", "both"]:

        ### Make scatterplots ###
        SAVE_DIR = f"plots/{args.dataset}_Simpson/{args.distribution}/{plot_metric_name}"
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

        if args.fitting_method == 'ODR':
            fig, ax = plt.subplots(figsize=(6,6))
            
            group_values = df[args.group].unique()
            group_values = sorted(group_values, key = float)
            for group_value in group_values:

                subgroup = df.loc[df[args.group] == group_value]
                y = subgroup[args.bleu_type].values
                x = subgroup[args.metric].values
                ax.scatter(x,y,s=35, label=group_value)
                plot_odr(x,y,ax)

            y = df[args.bleu_type].values
            x = df[args.metric].values
            plot_odr(x,y,ax,color='gray')

            xmin, xmax = df[args.metric].min(), df[args.metric].max()

            ax.set_xlabel(plot_metric_name, fontsize=18)
            ax.set_ylabel(plot_bleu_type_name, fontsize=18)
            #ax.set_ylim([-1,30])
            ax.set_xlim([xmin-(xmax-xmin)*0.1,xmax+(xmax-xmin)*0.1])
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f"{plot_metric_name} vs. {plot_bleu_type_name}", fontsize=18)
            legend = plt.legend(title=plot_group_name, bbox_to_anchor=(1.01, 1.0), loc='upper left', fontsize=14)
            plt.setp(legend.get_title(),fontsize=14)
            plt.savefig(
                os.path.join(SAVE_DIR, f"{args.bleu_type}_{plot_metric_name}_{args.group}_simpson_ODR.pdf"),
                bbox_inches='tight',
                #dpi=150,
                format='pdf',
            )

        elif args.fitting_method == 'LR':
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
                os.path.join(SAVE_DIR, f"{args.bleu_type}_{plot_metric_name}_{args.group}_simpson.pdf"),
                bbox_inches='tight',
                #dpi=150,
                format='pdf',
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
