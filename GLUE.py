import os
import transformers
import argparse
from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import wandb
from utils import preprocess_layers
import weightwatcher as ww
import pickle


# Global variables
metric = {}
    
def preprocess_function(examples, sentence1_key, sentence2_key, tokenizer):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if args.GLUE_task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric["new"].compute(predictions=predictions, references=labels)


def main():

    actual_task = "mnli" if args.GLUE_task == "mnli-mm" else args.GLUE_task
    dataset = load_dataset("glue", actual_task)
    metric["new"] = load_metric('glue', actual_task)
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

    sentence1_key, sentence2_key = task_to_keys[args.GLUE_task]
    if sentence2_key is None:
        print(f"Sentence: {dataset['train'][0][sentence1_key]}")
    else:
        print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
        print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

    preprocess_function_map = lambda x: preprocess_function(x, sentence1_key, sentence2_key, tokenizer)
    encoded_dataset = dataset.map(preprocess_function_map, batched=True)
    
    num_labels = 3 if args.GLUE_task.startswith("mnli") else 1 if args.GLUE_task=="stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=num_labels)

    preprocess_layers(args, model, model_type=args.model_checkpoint)

    metric_name = "pearson" if args.GLUE_task == "stsb" else "matthews_correlation" if args.GLUE_task == "cola" else "accuracy"
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
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_first_step=True
    )


    validation_key = "validation_mismatched" if args.GLUE_task == "mnli-mm" else "validation_matched" if args.GLUE_task == "mnli" else "validation"
    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

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
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_folder_step, num_labels=num_labels)

            trainer = Trainer(
                model,
                trainer_args,
                train_dataset=encoded_dataset["train"],
                eval_dataset=encoded_dataset[validation_key],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            final_eval = trainer.evaluate()
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
    parser.add_argument("--model_checkpoint", type=str, default="distilbert-base-uncased")
    parser.add_argument("--GLUE_task", type=str, default="cola", choices=["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"])

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
    parser.add_argument("--eval", action='store_true', help="evaluate f1 score after training")
    parser.add_argument("--eval_ww", action='store_true', help="evaluate weightwatcher after training")
    
    args = parser.parse_args()

    main()