import pickle
import numpy as np
from os import listdir
from os.path import isfile, join


def get_ww_layer_feature(ww_result, key):
    
    """
        Returns a particular key from the results
    """

    with open(ww_result, 'rb') as f:
        dict0 = pickle.load(f)
        feature_values = [dict0['details'][key][layer] for layer in dict0['details'][key].keys()]
        
        return feature_values
    
    
def get_alphas_Ds(ww_result):
    
    """
        Return both alphas and Ds
    """
    
    with open(ww_result, 'rb') as f:
        dict0 = pickle.load(f)
        
        assert 'alphas' in dict0 and 'Ds' in dict0, "This ww result does not contain alphas and/or Ds"
        
        alphas = dict0['alphas']
        Ds = dict0['Ds']
    
    return alphas, Ds


def get_ww_alpha(ww_result):
    
    """
        Return one single alpha from the summary dictionary
    """

    with open(ww_result, 'rb') as f:
        dict0 = pickle.load(f)
        alpha_value = dict0['summary']['alpha']
        
    return alpha_value


def compute_alpha_from_Ds(alphas, Ds, alpha_threshold=6, softmin=False, all_alpha=False, return_D=False, spectral_norms=None):
    
    temperature = 100
    layers = alphas.keys()
    # remove the last 10 points
    remove_last = 1
    
    alpha_layers = []
    D_layers = []
    for layer in layers:
        
        Ds_this_layer = Ds[layer]#[:-remove_last]
        alphas_this_layer = alphas[layer]#[:-remove_last]
        
        # remove the outliers
        mask = alphas_this_layer<alpha_threshold
        Ds_this_layer = Ds_this_layer[mask]
        alphas_this_layer = alphas_this_layer[mask]
        
        # This is the hardmin way
        #if len(Ds_this_layer)==0:
        #    alpha_layers.append(0)
        #    D_layers.append(0)
        #    continue
            
        min_D_index = np.argmin(Ds_this_layer)
        alpha_layers.append(alphas_this_layer[min_D_index])
        
        #print('-'*10)
        #print('layer', layer)
        #print('min_D_index', min_D_index)
        #print('alpha', alphas_this_layer[min_D_index])
        #print('size of the layer', len(alphas_this_layer))
        #print('D value', min(Ds_this_layer))
        
        # One possible solution is to directly add the 0-th entry
        #alpha_layers.append(alphas_this_layer[0])
        
        # How wide is the KS valley?
        #alpha_layers.append(np.sum(Ds_this_layer<0.15)/len(Ds_this_layer)*3)
        min_D = min(Ds_this_layer)
        D_layers.append(min_D)
        
        # This is the softmin way
        #WeightsD = np.exp(-temperature*Ds_this_layer)/sum(np.exp(-temperature*Ds_this_layer))
        #alpha_layers.append(alphas_this_layer.dot(WeightsD)) 
        
        # What about simple average?
        #alpha_layers.append(np.mean(alphas_this_layer))
        #alpha_layers.append(np.mean(Ds_this_layer*10))
        #alpha_layers.append(np.mean(alphas_this_layer[-10:]))
        
    # Get weighted alpha
    alpha_hat = alpha_layers*np.array(spectral_norms)
    #alpha = np.mean(alpha_hat)
    # Get spectral norm
    #alpha = np.mean(spectral_norms)
    # Get mean of alpha
    alpha = np.mean(alpha_layers)
    # Get min of alpha
    #alpha = np.min(alpha_layers)
    # Get mean of Ds
    #alpha = np.mean(D_layers)
    print('Got alpha from Ds')
    
    if return_D:
        print('Return D values')
        return D_layers
    if not all_alpha:
        #print('Return alpha values')
        #return alpha
        return alpha, {'alpha_hat': alpha_hat, 'alphas':alpha_layers, 'spectral_norms': spectral_norms}
    else:
        return alpha_layers


def get_ww_alpha_from_Ds(ww_result, all_alpha=False, return_D=False):
    
    """
        Return alpha using several different ways
    """
    
    alphas, Ds = get_alphas_Ds(ww_result)
    
    spectral_norms = get_ww_layer_feature(ww_result, 'log_spectral_norm')
    
    #print("Raw data for")
    #print(ww_result)
    
    if all_alpha:
        result = compute_alpha_from_Ds(alphas, Ds, all_alpha=all_alpha, return_D=return_D, spectral_norms=spectral_norms)
    else:
        result = compute_alpha_from_Ds(alphas, Ds, all_alpha=all_alpha, return_D=return_D, spectral_norms=spectral_norms)
    
    #raw_result_file = ww_result[:-24]+'results_raw_alphas.pkl'
    #if not os.path.exists(raw_result_file):
    #    with open(raw_result_file, 'wb') as f:
    #        pickle.dump(raw_results, f)
    
    return result


def get_ww_layer_alpha(ww_result, from_Ds=False):
    
    """
        Return the alpha values either from Ds or not
    """

    with open(ww_result, 'rb') as f:
        dict0 = pickle.load(f)
        if not from_Ds:
            alpha_values = [dict0['details']['alpha'][key] for key in dict0['details']['alpha'].keys()]
        else:
            alpha_values = get_ww_alpha_from_Ds(ww_result, all_alpha=True)
        
        return alpha_values
    
    
def get_ww_result(plot_result, ww_result, keys, get_alpha_from_Ds=False, min_alpha=False):
    
    """
        Return one list of results
        Some metrics, such as rand_distance, entropy, etc, are not in the "summary" dictionary
        In this case, one should simply get the average from all layers
    """

    #print(ww_result)
    with open(ww_result, 'rb') as f:
        dict0 = pickle.load(f)
        for key in keys:
            if key not in ['rand_num_spikes', 'rand_distance', 'entropy', 'exponent', 'D', 'sigma_var']:
                if get_alpha_from_Ds and key=='alpha':
                    #print('Get alpha from Ds')
                    alpha_value = get_ww_alpha_from_Ds(ww_result)
                    plot_result[key].append(alpha_value)
                elif key=='alpha' and min_alpha:
                    plot_result[key].append(dict0['details'][key].min())
                elif key=='num_spikes':
                    plot_result[key].append(dict0['details'][key].sum()/1000)
                else:
                    plot_result[key].append(dict0['summary'][key])
            else:
                layer_values = [dict0['details'][key][epoch] for epoch in dict0['details'][key].keys()]
                #if key == 'sigma_var':
                #    print(layer_values)
                if None in layer_values:
                    plot_result[key].append(0)
                    print("None encountered in calculation!")
                else:
                    plot_result[key].append(np.mean(layer_values))
                    

def get_ww_layer_metrics(ww_result, per_layer):
    
    """
        Return one list of metrics
    """
    
    if per_layer=='spikes':
        per_layer_feature = 'rand_num_spikes'
    else:
        per_layer_feature = per_layer
        
    with open(ww_result, 'rb') as f:
        dict0 = pickle.load(f)
        spike_values = [dict0['details'][per_layer_feature][key] for key in dict0['details'][per_layer_feature].keys()]
        
        return spike_values
    
    
def get_per_layer_curves_one_epoch(plot_result, ww_result, keys, layer_ids, from_Ds=False):
    
    """
        Return all the value-time curves for all layers
    """
    
    with open(ww_result, 'rb') as f:
        dict0 = pickle.load(f)
    
    if from_Ds:
        
        Ds = get_ww_alpha_from_Ds(ww_result, return_D=True)
        for layer_index, layer in enumerate(layer_ids):
            value = Ds[layer_index]
            plot_result['D'][layer].append(value)
        return
    
    for key in keys:
        for layer_index, layer in enumerate(layer_ids):
            value = dict0['details'][key][layer_index]
            plot_result[key][layer].append(value)
            
            
def get_all_layers(mypath):
    
    """
        Return the layer indices using the weightwatcher files
    """
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    layers = []
    for file in onlyfiles:
        file = file.split('.')
        if 'layer' in file[1]:
            layers.append(file[1][5:])
    all_layers = [int(x) for x in np.unique(layers)]
    all_layers.sort()
    print(all_layers)
    return all_layers


def get_all_layers_square(mypath):
    
    """
        Return the layer indices (only square layers) using the weightwatcher files
    """
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    layers = []
    for file in onlyfiles:
        file = file.split('.')
        if 'layer' in file[1] and 'mpfit1'==file[2]:
            layers.append(file[1][5:])
    all_layers = [int(x) for x in np.unique(layers)]
    all_layers.sort()
    print(all_layers)
    return all_layers
            