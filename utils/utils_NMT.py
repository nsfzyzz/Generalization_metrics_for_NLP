import os
import numpy as np
import pickle

def get_ckpt_folder(experiment_type, dataset, num_samples, width, training_type, directory_depth=0, 
    lr_factor=None, folder_suffix='', depth=None):
    
    """
        This function gets the checkpoint folder based on the type of experiment
    """

    data_folder = dataset
    if experiment_type=="sample":
        data_folder += f'_sample_{num_samples}_new{folder_suffix}'
    width_folder = f'w{width}'
    training_folder = training_type
    if experiment_type=="lr":
        training_folder += f'_lr_factor_{lr_factor}{folder_suffix}'
    
    if experiment_type=="depth":
        width_folder += f'_depth_{depth}{folder_suffix}'
    elif experiment_type=="width":
        width_folder += folder_suffix
    
    if directory_depth==0:
        directory_suffix = '.'
    if directory_depth==1:
        directory_suffix = '../'
        
    ckpt_folder = os.path.join(directory_suffix, f'../checkpoint/NMT_epochs/{data_folder}/{width_folder}/{training_folder}')

    return ckpt_folder


def get_epochs(args, num_samples=0):

    """
        This function gets the epochs based on the number of samples
    """
    if args.experiment_type == 'sample':
        
        epochs = range(21)

    elif args.experiment_type in ['width', 'lr', 'depth']:

        epochs = range(20)

    else: 
        
        raise ValueError("Not trained yet.")
        
    return epochs


def mk_metrics_folder(args, ckpt_folder):
    
    """
        This function makes a folder to store the ww metrics
    """

    print(ckpt_folder)
    assert os.path.exists(ckpt_folder)
    metric_folder = os.path.join(ckpt_folder, args.metric_folder)
    if not os.path.exists(metric_folder):
        os.mkdir(metric_folder)
    return


def get_experiment_folders_and_epochs(args):
    
    """
        This function gets checkpoint folders and epochs
    """

    ckpt_folders = []
    ckpt_epochs = []
    widths = []
    samples = []
    lr_factors = []
    depths = []

    if args.experiment_type == 'width':
        #width_list = [128, 256, 512, 768, 1024, 1536]
        width_list = [128, 192, 256, 384, 512]
        
        for width in width_list:
            ckpt_folder = get_ckpt_folder(args.experiment_type, args.dataset, 0, width, args.training_type, 
                                          directory_depth=args.directory_depth,
                                          folder_suffix=args.folder_suffix)
            if args.mkdir:
                if args.script_type=='train' and not os.path.exists(ckpt_folder):
                    os.makedirs(ckpt_folder)
                mk_metrics_folder(args, ckpt_folder)
            ckpt_folders.append(ckpt_folder)
            ckpt_epochs.append(get_epochs(args))
            widths.append(width)
            if args.dataset=='IWSLT':
                samples.append(0)
            else:
                samples.append(1280000)
            lr_factors.append(1)
            depths.append(6)

    if args.experiment_type == 'depth':
        if args.exclude_standard:
            depth_list = [2,3,4,5]
        else:
            depth_list = [2,3,4,5,None]
        width = args.IWSLT_width
        for depth in depth_list:
            ckpt_folder = get_ckpt_folder(args.experiment_type, args.dataset, 0, width, args.training_type, 
                                          directory_depth=args.directory_depth,
                                          folder_suffix=args.folder_suffix, depth=depth)
            if args.mkdir:
                if args.script_type=='train' and not os.path.exists(ckpt_folder):
                    os.makedirs(ckpt_folder)
                mk_metrics_folder(args, ckpt_folder)
            ckpt_folders.append(ckpt_folder)
            ckpt_epochs.append(get_epochs(args))
            widths.append(width)
            if args.dataset=='IWSLT':
                samples.append(0)
            else:
                samples.append(1280000)
            lr_factors.append(1)
            depths.append(depth)
    
    elif args.experiment_type == 'sample':
        if args.dataset == 'IWSLT':
            sample_list = [10000, 20000, 40000, 80000, 160000]
            width = args.IWSLT_width
        elif args.dataset == 'WMT':
            sample_list = [160000, 320000, 640000, 1280000]
            width = args.IWSLT_width
        for sample in sample_list:
            ckpt_folder = get_ckpt_folder(args.experiment_type, args.dataset, sample, width, args.training_type,
                                         directory_depth=args.directory_depth, folder_suffix=args.folder_suffix)
            if args.mkdir:
                if args.script_type=='train' and not os.path.exists(ckpt_folder):
                    os.makedirs(ckpt_folder)
                mk_metrics_folder(args, ckpt_folder)
            ckpt_folders.append(ckpt_folder)
            ckpt_epochs.append(get_epochs(args, num_samples=sample))
            widths.append(width)
            samples.append(sample)
            lr_factors.append(1)
            depths.append(6)

    elif args.experiment_type == 'lr':
        if args.exclude_standard:
            lr_factor_list = ['0.25', '0.375', '0.5', '0.75', '2']
        else:
            lr_factor_list = ['0.25', '0.375', '0.5', '0.75', None, '2']
        for lr_factor in lr_factor_list:
            width = args.IWSLT_width
            ckpt_folder = get_ckpt_folder(args.experiment_type, args.dataset, 0, width, args.training_type, lr_factor=lr_factor,
                                         directory_depth=args.directory_depth, folder_suffix=args.folder_suffix)
            print(ckpt_folder)
            if args.mkdir:
                if args.script_type=='train' and not os.path.exists(ckpt_folder):
                    os.makedirs(ckpt_folder)
                mk_metrics_folder(args, ckpt_folder)
            ckpt_folders.append(ckpt_folder)
            ckpt_epochs.append(get_epochs(args))
            widths.append(width)
            if args.dataset=='IWSLT':
                samples.append(0)
            else:
                samples.append(1280000)
            lr_factors.append(lr_factor)
            depths.append(6)

    return ckpt_folders, ckpt_epochs, widths, samples, lr_factors, depths



def get_val_loss(plot_result, loss_result):
    
    """
        This function gets validation loss and saves in the plot_result dictionary
    """

    plot_result['validation_loss'] = [] 
    with open(loss_result, 'r') as file:
        lines = file.readlines()
        for x in lines:
            x = x.split(' ')
            if 'validation' and 'loss' in x:
                plot_result['validation_loss'].append(float(x[-1]))

                
def get_bleu_score(plot_result, loss_result, divide):
    
    """
        This function gets the bleu score and saves in the plot_result di
    """

    plot_result['bleu_score'] = [] 
    with open(loss_result, 'r') as file:
        lines = file.readlines()
        for x in lines:
            x = x.split(' ')
            if 'BLEU' in x and 'at' in x and 'score' in x:
                plot_result['bleu_score'].append(float(x[-1])*100/divide)
                

def get_bleu_gap(plot_result, loss_result, divide):
    
    """
        This function gets the bleu score and saves in the plot_result di
    """

    dict0 = pickle.load(open(loss_result, 'rb'))
    epochs = list(dict0.keys())
    epochs.sort()
    plot_result['bleu_score'] = [dict0[epoch]['bleu_score_generalization']/divide for epoch in epochs]

                
def get_train_loss(plot_result, loss_result):

    plot_result['training_loss'] = [] 
    with open(loss_result, 'r') as file:
        lines = file.readlines()
        for x in lines:
            x = x.split(' ')
            if 'training' and 'loss=' in x:
                plot_result['training_loss'].append(float(x[-1]))
        plot_result['training_loss'] = smooth(plot_result['training_loss'], box_pts=20)
                
                
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth