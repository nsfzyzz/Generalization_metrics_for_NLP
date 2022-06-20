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
            number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
            number_of_layers=args.num_layers,
            dropout_probability=BASELINE_MODEL_DROPOUT_PROB
        )

    # Compute metrics for all epochs
    ww_metrics = {}     # key: epoch, value: results dict
    EPOCHS = 20
    wandb.init(name = args.ckpt + '_ww')

    for epoch in range(1, EPOCHS+1):
        print(f"\nEPOCH {epoch}")
        ckpt = torch.load(os.path.join(args.ckpt,f"net_epoch_{epoch}.ckpt"), map_location='cpu')
        baseline_transformer.load_state_dict(ckpt["state_dict"])

        print("Start weight watcher analysis.")

        watcher = ww.WeightWatcher(model=baseline_transformer)
        if args.continuous_estimate == 0:
            details = watcher.analyze(
                mp_fit=args.mp_fit,
                randomize=args.randomize,
                plot=args.save_plot,
                savefig=args.result,
                distribution=args.distribution,     # distribution is only for WeightWatcher2
            )
            summary = watcher.get_summary(details)
            results = {'details':details, 'summary':summary}
            ww_metrics[epoch] = results

            wandb.log(summary)

        # Don't use this
        else:
            if not args.heuMax:
                details, alphas_results, Ds_results = watcher.analyze(mp_fit=args.mp_fit, randomize=args.randomize, plot=args.save_plot, 
                        savefig=args.result, continuous_estimate=float(args.continuous_estimate), fix_finger_histogram=args.fix_finger_histogram)
            else:
                details, alphas_results, Ds_results = watcher.analyze(mp_fit=args.mp_fit, randomize=args.randomize, plot=args.save_plot, 
                        savefig=args.result, continuous_estimate=float(args.continuous_estimate), 
                        heuristic_xmax=True, heuristic_xmax_factor=args.heuMax_factor, fix_finger_histogram=args.fix_finger_histogram)
            summary = watcher.get_summary(details)

            results = {'details':details, 'summary':summary, 'alphas': alphas_results, 'Ds': Ds_results}
            ww_metrics[epoch] = results
    
    # Write all results into one file
    with open(os.path.join(args.result, args.result_suffix), 'wb') as f:
        pickle.dump(ww_metrics, f)

    print("Experiment finished. Save and exit.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path of checkpoint")
    parser.add_argument("result", type=str, help="path to save result")
    parser.add_argument("--result-suffix", type=str, default='results.pkl')
    parser.add_argument("--width", type=int, help="embedding dimension", default=64)
    parser.add_argument("--dataset", type=str, help="dataset", choices=['IWSLT', 'WMT'], default='IWSLT')
    parser.add_argument("--batch_size", type=int, help="batch size to create dataset", default=1500)
    parser.add_argument("--num-samples", type=int, help="number of samples", default=0)
    parser.add_argument("--save-plot", action='store_true', help="save plot of the weightwatcher results")
    parser.add_argument("--mp-fit", action='store_true', help="fitting the model using MP Fit.")
    parser.add_argument("--randomize", action='store_true', help="use randomized matrix to check correlation trap.")
    parser.add_argument("--not-rescale", action='store_true', help="do not use rescale to solve a bug")
    parser.add_argument("--continuous-estimate", default=0, help="use continuous estimate")
    parser.add_argument("--heuMax", action='store_true', help="Use heuristic Max")
    parser.add_argument("--heuMax-factor", default=0.8, type=float, help="use continuous estimate")
    parser.add_argument("--fix-finger-histogram", action='store_true', help="use finger fix")
    # distribution is only for WeightWatcher2
    parser.add_argument("--distribution", choices=["truncated_power_law", "power_law", "lognormal", "exponential"])
    parser.add_argument("--num-layers", type=int, help="number of Transformer layers", default=6)

    #parser.add_argument("--negative-lambda", action='store_true', default=False)

    args = parser.parse_args()
    print(ww.__file__)

    print("Arguments for the experiment.")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    main(args)

