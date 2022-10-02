import torch
import weightwatcher as ww
import argparse
import os
import pickle
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders
from utils.constants import *
import wandb

def main(args):
    
    device = torch.device("cpu")
    subsampling = args.num_samples!=0

    # Get Transformer model
    print("Load transformer model.")

    _, _, src_field_processor, trg_field_processor = get_data_loaders(
        './data',
        'G2E',
        args.dataset,
        args.batch_size,
        device,
        subsampling=subsampling,
        num_samples=args.num_samples)

    #pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    baseline_transformer = Transformer(
            model_dimension=args.width,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            number_of_heads=args.num_heads,
            number_of_layers=args.num_layers,
            dropout_probability=BASELINE_MODEL_DROPOUT_PROB
        )

    # Compute metrics for all epochs
    ww_metrics = {}     # key: epoch, value: results dict
    wandb.init(name = args.ckpt + '_ww')

    if args.distribution == 'truncated_power_law':
        distribution = 'E_TPL'
    elif args.distribution == 'power_law':
        distribution = 'PL'
    elif args.distribution == 'exponential':
        distribution = 'EXP'

    for epoch in range(args.starting_epoch, args.num_epochs+1):
        print(f"\nEPOCH {epoch}")
        ckpt = torch.load(os.path.join(args.ckpt,f"net_epoch_{epoch}.ckpt"), map_location='cpu')
        baseline_transformer.load_state_dict(ckpt["state_dict"])

        print("Start weight watcher analysis.")

        watcher = ww.WeightWatcher(model=baseline_transformer)
        details = watcher.analyze(
            mp_fit=args.mp_fit,
            randomize=args.randomize,
            plot=args.save_plot,
            savefig=args.result,
            fit=distribution,     # distribution is only for WeightWatcher2
        )
        summary = watcher.get_summary(details)
        results = {'details':details, 'summary':summary}
        ww_metrics[epoch] = results

        wandb.log(summary)
    
    # Write all results into one file
    #with open(os.path.join(args.result, args.result_suffix), 'wb') as f:
    #    pickle.dump(ww_metrics, f)

    print("Experiment finished. Save and exit.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path of checkpoint")
    parser.add_argument("result", type=str, help="path to save result")
    parser.add_argument("--result-suffix", type=str, default='results.pkl')
    parser.add_argument("--width", type=int, help="embedding dimension", default=64)
    parser.add_argument("--dataset", type=str, help="dataset", choices=['IWSLT', 'WMT'], default='WMT')
    parser.add_argument("--batch_size", type=int, help="batch size to create dataset", default=1500)
    parser.add_argument("--num-samples", type=int, help="number of samples", default=0)
    parser.add_argument("--save-plot", action='store_true', help="save plot of the weightwatcher results")
    parser.add_argument("--mp-fit", action='store_true', help="fitting the model using MP Fit.")
    parser.add_argument("--randomize", action='store_true', help="use randomized matrix to check correlation trap.")
    parser.add_argument("--distribution", choices=["truncated_power_law", "power_law", "lognormal", "exponential"])
    parser.add_argument("--num-layers", type=int, help="number of Transformer layers", default=6)
    parser.add_argument("--num-epochs", type=int, help="number of epochs", default=20)
    parser.add_argument("--starting-epoch", type=int, help="The starting epoch number", default=1)
    parser.add_argument("--num-heads", type=int, help="number of Transformer heads", default=BASELINE_MODEL_NUMBER_OF_HEADS)

    #parser.add_argument("--negative-lambda", action='store_true', default=False)

    args = parser.parse_args()
    print(ww.__file__)

    print("Arguments for the experiment.")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    main(args)

