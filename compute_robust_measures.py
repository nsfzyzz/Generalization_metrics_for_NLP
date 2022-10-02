import torch
import argparse
import os
import pickle
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders
from utils.constants import *
import wandb
from robust_measures import get_all_measures
from utils.utils_CKA import *


class fake_dataloader:
    def __init__(self, dataset):
        self.dataset = dataset


def main(args):
    
    if not args.calculate_margin and not args.calculate_pac_bayes and not args.test_bleu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    subsampling = args.num_samples!=0

    # Get Transformer model
    print("Load transformer model.")

    train_token_ids_loader, _,  src_field_processor, trg_field_processor = get_data_loaders(
        './data',
        'G2E',
        args.dataset,
        args.batch_size,
        device,
        subsampling=subsampling,
        num_samples=args.num_samples)

    if args.dataset=='IWSLT':
        dataset_len = 200000
    elif args.dataset == 'WMT':
        dataset_len = 4500000
    if args.num_samples!=0:
        dataset_len = args.num_samples
    fake_NMT_loader = fake_dataloader(dataset=[0]*dataset_len)

    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    # Load initialized model
    baseline_transformer_init = Transformer(
            model_dimension=args.width,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            number_of_heads=args.num_heads,
            number_of_layers=args.num_layers,
            dropout_probability=BASELINE_MODEL_DROPOUT_PROB
        ).to(device)

    ckpt_epoch = os.path.join(args.ckpt, f"net_epoch_{args.starting_epoch}.ckpt")
    ckpt = torch.load(ckpt_epoch, map_location='cpu')
    baseline_transformer_init.load_state_dict(ckpt["state_dict"])
    baseline_transformer_init.eval()

    wandb.init(name = args.ckpt + '_eval_measure')

    final_evals = {}
    for epoch in range(1, 1+args.num_epochs):
        
        all_complexities = {}
        print(f"Loading the checkpoint for epoch {epoch}.")
        baseline_transformer = Transformer(
            model_dimension=args.width,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            number_of_heads=args.num_heads,
            number_of_layers=args.num_layers,
            dropout_probability=BASELINE_MODEL_DROPOUT_PROB
        ).to(device)

        baseline_transformer_path_norm = Transformer(
            model_dimension=args.width,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            number_of_heads=args.num_heads,
            number_of_layers=args.num_layers,
            dropout_probability=BASELINE_MODEL_DROPOUT_PROB,
            customize_layer_norm=True
        )

        ckpt_epoch = os.path.join(args.ckpt, f"net_epoch_{epoch}.ckpt")
        ckpt = torch.load(ckpt_epoch, map_location='cpu')
        baseline_transformer.load_state_dict(ckpt["state_dict"])
        baseline_transformer.eval()
            
        baseline_transformer_path_norm.load_state_dict(ckpt["state_dict"])
        baseline_transformer_path_norm.eval()

        if args.calculate_margin:
            measure_loader = train_token_ids_loader
        else:
            measure_loader = fake_NMT_loader

        print("Start analysis on different types of measures.")

        all_complexities = get_all_measures(baseline_transformer, 
                baseline_transformer_init,
                measure_loader, 
                None,
                seed=2021, 
                no_pac_bayes=not args.calculate_pac_bayes, 
                no_margin=not args.calculate_margin,
                no_basics=False,
                no_path_norm=False,
                no_CKA=False,
                path_norm_transformer=baseline_transformer_path_norm, 
                pad_token_id=pad_token_id,
                trg_vocab_size=trg_vocab_size,
                pacbayes_depth=8)
        final_evals[epoch] = all_complexities

    wandb.log(all_complexities)
    pickle.dump(final_evals, open( os.path.join(args.ckpt, args.result_suffix), "wb" ) )

    print("Experiment finished. Save and exit.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path of checkpoint")
    parser.add_argument("--result_suffix", type=str, default='robust_measures.pkl', help="name of result")
    parser.add_argument('--starting-epoch', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument("--width", type=int, help="embedding dimension", default=64)
    parser.add_argument("--dataset", type=str, help="dataset", choices=['IWSLT', 'WMT'], default='IWSLT')
    parser.add_argument("--batch_size", type=int, help="batch size to create dataset", default=1500)
    parser.add_argument("--num-samples", type=int, help="number of samples", default=0)
    parser.add_argument("--calculate_margin", action='store_true')
    parser.add_argument("--calculate_pac_bayes", action='store_true')
    parser.add_argument("--num-layers", type=int, help="number of Transformer layers", default=6)
    parser.add_argument("--num-heads", type=int, help="number of Transformer heads", default=BASELINE_MODEL_NUMBER_OF_HEADS)
    
    args = parser.parse_args()
    
    print("Arguments for the experiment.")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    main(args)

