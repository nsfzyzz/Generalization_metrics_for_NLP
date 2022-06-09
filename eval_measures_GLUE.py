import os
import transformers
import argparse
from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import wandb
from utils import preprocess_layers
import weightwatcher as ww
import pickle
from measures import get_all_measures, eval_acc
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

def preprocess_function(examples, sentence1_key, sentence2_key, tokenizer):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


def main():
    
    wandb.init(name = args.ckpt_folder + '_eval_measure')

    actual_task = "mnli" if args.GLUE_task == "mnli-mm" else args.GLUE_task
    dataset = load_dataset("glue", actual_task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    # First, get the initial model
    num_labels = 3 if args.GLUE_task.startswith("mnli") else 1 if args.GLUE_task=="stsb" else 2
    ckpt_folder_step = os.path.join(args.ckpt_folder, f"checkpoint-0")
    init_model = AutoModelForSequenceClassification.from_pretrained(ckpt_folder_step, num_labels=num_labels)

    sentence1_key, sentence2_key = task_to_keys[args.GLUE_task]
    if sentence2_key is None:
        print(f"Sentence: {dataset['train'][0][sentence1_key]}")
    else:
        print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
        print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

    preprocess_function_map = lambda x: preprocess_function(x, sentence1_key, sentence2_key, tokenizer)
    encoded_dataset = dataset.map(preprocess_function_map, batched=True)
    
    # This part is used to pad the dataset
    trainer_args = TrainingArguments(
        "test-glue"
    )
    validation_key = "validation_mismatched" if args.GLUE_task == "mnli-mm" else "validation_matched" if args.GLUE_task == "mnli" else "validation"
    trainer = Trainer(
        init_model,
        trainer_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    train_dataloader = DataLoader(encoded_dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)

    final_evals = {}
    steps = np.arange(0, args.max_steps+1, args.save_steps)

    for step in steps:

        ckpt_folder_step = os.path.join(args.ckpt_folder, f"checkpoint-{step}")
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_folder_step, num_labels=num_labels).cuda()

        acc = eval_acc(model, train_dataloader)
        all_complexities = get_all_measures(model, init_model, train_dataloader, acc, seed=2021, 
                                            no_pac_bayes=True, no_margin=False)

        final_evals[step] = all_complexities
        wandb.log(all_complexities)

    pickle.dump(final_evals, open( os.path.join(args.ckpt_folder, "robust_measures.pkl"), "wb" ) )


if __name__ == "__main__":

    print("Experiment starting. Transformer version is:")
    print(transformers.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder", type=str, default="tmp_SQuAD")
    parser.add_argument("--model_checkpoint", type=str, default="albert-base-v2")
    parser.add_argument("--GLUE_task", type=str, default="sst2", help='no cola and stsb because they are not classification',
        choices=["mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=200)
    
    args = parser.parse_args()

    main()