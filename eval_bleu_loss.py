'''
Computes training and validation BLEU scores and losses.
'''

import torch
import torch.nn as nn
import argparse
import os, json, re

from models.definitions.transformer_model import Transformer
import utils.utils as utils
from utils.constants import *
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
from utils.optimizers_and_distributions import LabelSmoothingDistribution

def eval_loss(model, dataloader, max_batches):
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # gives better BLEU score than "mean"
    label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, DEVICE)
    
    total_loss, n_batches = 0, 0

    with torch.no_grad():
        for batch_idx, token_ids_batch in enumerate(dataloader):
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, DEVICE)

            # log because the KL loss expects log probabilities (just an implementation detail)
            predicted_log_distributions = model(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)  # these are regular probabilities

            loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)
            loss.item()
            total_loss += loss.item()
            n_batches += 1

            if batch_idx >= max_batches:
                break

    return total_loss / n_batches       # average loss per batch

def eval_bleu(model, dataloader, trg_field_processor, max_batches):
    bleu_score = utils.calculate_bleu_score(
        model,
        dataloader,
        trg_field_processor,
        max_batch=max_batches,
    )
    return bleu_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--dataset", type=str, default='WMT')

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Extract info from checkpoint ###
    NUM_SAMPLES = re.search("sample(\d+)", args.checkpoint_dir)
    NUM_SAMPLES = int(NUM_SAMPLES.group(1))
    print(f"NUM_SAMPLES: {NUM_SAMPLES}")
    
    DEPTH = re.search("depth(\d+)", args.checkpoint_dir)
    DEPTH = int(DEPTH.group(1))
    print(f"DEPTH: {DEPTH}")
    
    if args.dataset == 'WMT':
        id_data, ood_data = DatasetType.WMT14.name, DatasetType.IWSLT.name
    elif args.dataset == 'IWSLT':
        ood_data, id_data = DatasetType.WMT14.name, DatasetType.IWSLT.name
    else:
        raise NameError('Dataset not implemented yet.')
    
    ood_train_token_ids_loader, ood_val_token_ids_loader, _, _ = get_data_loaders(
        DATA_DIR_PATH,
        LanguageDirection.G2E.name,
        ood_data,
        1500,
        DEVICE,
        subsampling=True,
        num_samples=NUM_SAMPLES,
        ood=True,
    )

    id_train_token_ids_loader, id_val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        DATA_DIR_PATH,
        LanguageDirection.G2E.name,
        id_data,
        1500,
        DEVICE,
        subsampling=True,
        num_samples=NUM_SAMPLES,
        ood=False,
    )

    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    model = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=DEPTH,
        dropout_probability=0.0,
    )
    ###
    
    NUM_EPOCHS = 20
    output = ""
    
    # Compute metrics for epochs 1-20
    for EPOCH in range(1, NUM_EPOCHS+1):
        ckpt_file = os.path.join(args.checkpoint_dir, f"net_epoch_{EPOCH}.ckpt")
        model.load_state_dict(
            torch.load(ckpt_file, map_location=DEVICE)["state_dict"]
        )
        model.to(DEVICE)
        model.eval()
        
        metrics = {}
        print(f"Getting metrics for epoch {EPOCH}...")
        # Compute training BLEU/loss
        metrics[f'epoch{EPOCH}_id_train_loss'] = eval_loss(model, id_train_token_ids_loader, args.max_batches)
        metrics[f'epoch{EPOCH}_id_train_bleu_score'] = eval_bleu(model, id_train_token_ids_loader, trg_field_processor, args.max_batches)
        # Compute validation BLEU/loss
        metrics[f'epoch{EPOCH}_id_val_loss'] = eval_loss(model, id_val_token_ids_loader, args.max_batches)
        metrics[f'epoch{EPOCH}_id_bleu_score'] = eval_bleu(model, id_val_token_ids_loader, trg_field_processor, args.max_batches)
        metrics[f'epoch{EPOCH}_ood_val_loss'] = eval_loss(model, ood_val_token_ids_loader, args.max_batches)
        metrics[f'epoch{EPOCH}_ood_bleu_score'] = eval_bleu(model, ood_val_token_ids_loader, trg_field_processor, args.max_batches)
        print(metrics)
        output += (json.dumps(metrics) + "\n")

    # Write entire output to file at once
    with open(os.path.join(args.checkpoint_dir, "bleu_loss.jsonl"), "w+") as file:
        file.write(output)