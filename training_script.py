"""
Adapted from "Pytorch Original Transformer" by Aleksa Gordić
https://github.com/gordicaleksa/pytorch-original-transformer
"""

import argparse
import time


import torch
from torch import nn
from torch.optim import Adam

from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *

num_of_trg_tokens_processed = 0
bleu_scores = []
global_train_step, global_val_step = [0, 0]
best_val_loss = None
import wandb

def get_train_val_loop(baseline_transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time_start, training_config, save_ckpt=True):

    def train_val_loop(is_train, token_ids_loader, epoch):
                
        global num_of_trg_tokens_processed, global_train_step, global_val_step, best_val_loss

        if is_train:
            baseline_transformer.train()
        else:
            baseline_transformer.eval()

        device = next(baseline_transformer.parameters()).device
        
        validation_loss = 0
        val_step = 0
        
        
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)

            predicted_log_distributions = baseline_transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)

            if is_train:
                custom_lr_optimizer.zero_grad()

            loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)

            if is_train:
                loss.backward()
                custom_lr_optimizer.step()

            if is_train:
                global_train_step += 1
                num_of_trg_tokens_processed += num_trg_tokens

                training_loss = loss.item()
                if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                    print(f'Transformer training: time elapsed= {(time.time() - time_start):.2f} [s] '
                          f'| epoch={epoch + 1} | batch= {batch_idx + 1} '
                          f'| target tokens/batch= {num_of_trg_tokens_processed / training_config["console_log_freq"]} '
                          f'| global training step= {global_train_step} '
                          f'| training loss= {training_loss}')

                    num_of_trg_tokens_processed = 0

                if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0 and batch_idx == 0:
                    
                    ckpt_name = os.path.join(args.checkpoint_path, f'net_epoch_{(epoch+1)}{args.checkpoint_suffix}.ckpt')
                    torch.save(utils.get_training_state(training_config, baseline_transformer), ckpt_name)
            else:
                val_step += 1
                validation_loss += loss.item()

        if not is_train:
            
            validation_loss = validation_loss/val_step
            
            print('-'*30)
            print(f'Validation loss at epoch={epoch + 1} is {validation_loss}')
            print('-'*30)

            wandb.log({'Validation_loss': validation_loss})
            
            if save_ckpt and (not best_val_loss or best_val_loss > validation_loss):
                best_val_loss = validation_loss
                print("The newly trained model has better validation loss. Save this model!")
                
                ckpt_name = os.path.join(args.checkpoint_path, f'net_exp_{args.exp_ind}{args.checkpoint_suffix}_best.ckpt')
                torch.save(utils.get_training_state(training_config, baseline_transformer), ckpt_name)


    return train_val_loop

def train_transformer(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: Prepare data loaders
    train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        training_config['dataset_path'],
        training_config['language_direction'],
        training_config['dataset_name'],
        training_config['batch_size'],
        device,
        subsampling=args.subsampling,
        num_samples=args.num_samples)

    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    # Step 2: Prepare the model (original transformer) and push to GPU
    baseline_transformer = Transformer(
        model_dimension=args.embedding_dimension,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=args.num_heads,
        number_of_layers=args.num_layers,
        dropout_probability=args.dropout,
    ).to(device)

    # Step 3: Prepare other training related utilities
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # gives better BLEU score than "mean"

    label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)

    custom_lr_optimizer = CustomLRAdamOptimizer(
                Adam(baseline_transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
                args.embedding_dimension,
                training_config['num_warmup_steps'],
                lr_inverse_dim=args.lr_inverse_dim,
                constant_lr =args.constant_lr,
                lr_factor = args.lr_factor
            )

    wandb.init(name = args.checkpoint_path + '_train')

    train_val_loop = get_train_val_loop(baseline_transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time.time(), training_config)

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    ckpt_name = os.path.join(args.checkpoint_path, f'net_epoch_0{args.checkpoint_suffix}.ckpt')
    torch.save(utils.get_training_state(training_config, baseline_transformer), ckpt_name)
    
    with torch.no_grad():
        train_val_loop(is_train=False, token_ids_loader=val_token_ids_loader, epoch=-1)

        bleu_score = utils.calculate_bleu_score(baseline_transformer, val_token_ids_loader, trg_field_processor)
        print('-'*30)
        print(f'BLEU score at epoch=0 is {bleu_score}')
        print('-'*30)
        wandb.log({'BLEU_score': bleu_score})


    for epoch in range(training_config['num_of_epochs']):

        train_val_loop(is_train=True, token_ids_loader=train_token_ids_loader, epoch=epoch)

        with torch.no_grad():
            train_val_loop(is_train=False, token_ids_loader=val_token_ids_loader, epoch=epoch)

            bleu_score = utils.calculate_bleu_score(baseline_transformer, val_token_ids_loader, trg_field_processor)
            print('-'*30)
            print(f'BLEU score at epoch={epoch + 1} is {bleu_score}')
            print('-'*30)
            wandb.log({'BLEU_score': bleu_score})
            
        if global_train_step > args.max_gradient_steps:
            print("Enough training! Saving model and exit.")
            
            ckpt_name = os.path.join(args.checkpoint_path, f'net_exp_{args.exp_ind}{args.checkpoint_suffix}.ckpt')
            torch.save(utils.get_training_state(training_config, baseline_transformer), ckpt_name)
            break
            
        else:
            print(f"global training step {global_train_step} is not reached. Continue.")

if __name__ == "__main__":

    num_warmup_steps = 4000
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=200)
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Data related args
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    
    # Some parameters to cover the previous training setup
    parser.add_argument("--subsampling", help="subsample data", action="store_true")
    parser.add_argument("--lr-inverse-dim", help="use the heuristic of inverse learning rate proportional to the model dimension", action="store_true")
    parser.add_argument("--constant-lr", help="use constant learning rate", action="store_true")
    
    parser.add_argument("--num-samples", type=int, help="number of samples (src-tar pairs)", default=1000)
    parser.add_argument("--checkpoint-path", type=str, help="checkpoint model saving path", default=CHECKPOINTS_PATH)
    parser.add_argument("--checkpoint-suffix", type=str, help="checkpoint model saving name", default="")
    parser.add_argument("--exp-ind", type=int, help="index of the training", default=0)
    parser.add_argument("--max-gradient-steps", type=int, help="number of gradient steps to train", default=80000)
    parser.add_argument("--embedding-dimension", type=int, help="the dimension to save a checkpoint", default=BASELINE_MODEL_DIMENSION)
    parser.add_argument("--lr-factor", type=float, help="factor to adjust the inverse dim lr", default=1.0)
    parser.add_argument("--dropout", type=float, help="dropout probability", default=0.0)
    parser.add_argument("--num-layers", type=int, help="number of Transformer layers", default=6)
    parser.add_argument("--num-heads", type=int, help="number of Transformer layers", default=BASELINE_MODEL_NUMBER_OF_HEADS)
    
    # Some parameters to change the Sharpness transform
    parser.add_argument("--sharpness-transform", help="apply sharpness transform?", action="store_true")
    parser.add_argument("--sharpness-perbatch", help="apply sharpness transform to each batch?", action="store_true")
    parser.add_argument("--sharpness-frequency", type=int, help="how many batches should we apply the transform", default=100)

    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['num_warmup_steps'] = num_warmup_steps

    print(args.checkpoint_path)

    # Train the original transformer model
    train_transformer(training_config)
