import glob
import os
import pickle
import numpy as np

datasets=['WMT14', 'IWSLT', 'WMT14'] 
size_params = ['depth', 'depth', 'width']
adjust_measures_suffixs = ['normalized_by_samples', 'not_normalized_by_samples']
bleu_types = ['id_bleu', 'id_bleu_gap']

for bleu_type in bleu_types:
    for dataset, size_param in zip(datasets, size_params):
        metric_folders = glob.glob(f"results/{dataset}_Simpson/*/*")

        for adjust_measures_suffix in adjust_measures_suffixs:
            if size_param == 'depth':
                groups = ['sample', 'lr', 'depth']
            elif size_param == 'width':
                groups = ['sample', 'lr', 'width']
            for group in groups:
                
                correlation_file = f'plot_results_test_Simpson_{dataset}_size_param_{size_param}_individual_param_{group}_{bleu_type}_{adjust_measures_suffix}.pkl'
                print(f"Generating {correlation_file}.")

                results = {}

                for path in metric_folders:
                    metric = os.path.basename(path)
                    corr_result_file = f"corr_{bleu_type}_{dataset}_size_param_{size_param}_{adjust_measures_suffix}.pkl"
                    results_this_metric = pickle.load(open(os.path.join(path, corr_result_file), 'rb'))

                    if metric in ['alpha_weighted', 'log_alpha_norm'] and 'TPL' in path:
                        continue

                    results[metric] = results_this_metric[group]

                    if bleu_type == 'id_bleu':
                        #print('negate the results!')
                        # Negate the correlation because we want the metrics to be negatively correlated
                        results[metric] = [-x for x in results[metric] if not np.isnan(x)]
                        #print(results[metric])
                pickle.dump(results, open(f'results/{correlation_file}', 'wb'))