import weightwatcher as ww
import transformers
import argparse
from datasets import load_dataset, ClassLabel
import random
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
import math
from utils import preprocess_layers
import os
from transformers import DataCollatorForLanguageModeling
import wandb
import numpy as np
import pickle


def tokenize_function(examples, tokenizer):

    return tokenizer(examples["text"])


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():

    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    tokenize_function_map = lambda x: tokenize_function(x, tokenizer)
    tokenized_datasets = datasets.map(tokenize_function_map, batched=True, num_proc=4, remove_columns=["text"])

    # block_size = tokenizer.model_max_length
    block_size = 128

    group_texts_map = lambda x: group_texts(x, block_size)
    lm_datasets = tokenized_datasets.map(
        group_texts_map,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model_checkpoint)
    if not args.pretrain:
        # randomize all the layers of the model
        for name, param in model.named_modules():
            if isinstance(param, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                model._init_weights(param)
                print(f"Randomizing the weights of layer {name}")

    model_name = args.model_checkpoint.split("/")[-1]
    
    trainer_args = TrainingArguments(
        args.ckpt_folder,
        evaluation_strategy = "steps",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        logging_steps=args.save_steps,
        save_steps=args.save_steps,
        weight_decay=args.weight_decay,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )
    preprocess_layers(args, model, model_type=args.model_checkpoint)

    # Save model for later evaluation
    trainer.save_model(os.path.join(args.ckpt_folder, 'checkpoint-0'))
    
    trainer.train()
    wandb.finish()

    if args.eval:

        wandb.init(name = args.ckpt_folder + '_eval')

        final_evals = {}
        steps = np.arange(0, args.max_steps+1, args.save_steps)
        for step in steps:

            ckpt_folder_step = os.path.join(args.ckpt_folder, f"checkpoint-{step}")
            model = AutoModelForMaskedLM.from_pretrained(ckpt_folder_step)

            trainer = Trainer(
                model=model,
                args=trainer_args,
                train_dataset=lm_datasets["train"],
                eval_dataset=lm_datasets["validation"],
                data_collator=data_collator,
            )
            final_eval = trainer.evaluate()
            final_eval['perplexity'] = math.exp(final_eval['eval_loss'])

            final_evals[step] = final_eval
            wandb.log(final_eval)

            if args.eval_ww:
                watcher = ww.WeightWatcher(model=model)
                details = watcher.analyze()
                summary = watcher.get_summary(details)
                wandb.log({"alpha": summary['alpha']})

                ww_results = {"details": details, "summary": summary}
                pickle.dump(ww_results, open( os.path.join(ckpt_folder_step, "ww_results.pkl"), "wb" ) )

        pickle.dump(final_evals, open( os.path.join(args.ckpt_folder, "final_eval.pkl"), "wb" ) )


if __name__ == "__main__":

    print("Experiment starting. Transformer version is:")
    print(transformers.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder", type=str, default="tmp_SQuAD")
    parser.add_argument("--model_checkpoint", type=str, default="distilroberta-base")
    parser.add_argument("--pretrain", action='store_true', help="use pretrained model")
    parser.add_argument("--train_one_layer", action='store_true', help="finetune only one layer")
    parser.add_argument("--randomize_layers", action='store_true', help="randomize some layers")
    parser.add_argument("--randomize_layers_num", type=int, default=1, 
                                    help="number of randomized layers")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--eval", action='store_true', help="evaluate some metrics after training")
    parser.add_argument("--eval_ww", action='store_true', help="evaluate weightwatcher after training")
    parser.add_argument("--max_length", type=int, default=384, 
                        help="The maximum length of a feature (question and context)")
    parser.add_argument("--doc_stride", type=int, default=128,
                help="The authorized overlap between two part of the context when splitting it is needed.")

    args = parser.parse_args()

    main()