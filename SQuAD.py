#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import collections
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, load_metric
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import pickle
import wandb
import weightwatcher as ww
from utils import preprocess_layers


def set_seed(seed = 0):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_train_features(examples, tokenizer, pad_on_right):
    
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, pad_on_right):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(examples, features, tokenizer, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not args.squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


def compute_SQuAD_metrics(datasets, trainer, tokenizer, pad_on_right):
    
    validation_features = datasets["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer, pad_on_right),
        batched=True,
        remove_columns=datasets["validation"].column_names
    )

    raw_predictions = trainer.predict(validation_features)
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    max_answer_length = 30

    final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, tokenizer, raw_predictions.predictions)
    metric = load_metric("squad_v2" if args.squad_v2 else "squad")

    if args.squad_v2:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    final_eval = metric.compute(predictions=formatted_predictions, references=references)
    return final_eval


def eval_f1(tokenized_datasets, tokenizer, model, datasets, pad_on_right):

    data_collator = default_data_collator
    eval_args = TrainingArguments(
        f"eval",
    )
    trainer = Trainer(
        model,
        eval_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return compute_SQuAD_metrics(datasets, trainer, tokenizer, pad_on_right)


def main():

    datasets = load_dataset("squad_v2" if args.squad_v2 else "squad")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    pad_on_right = tokenizer.padding_side == "right"
    tokenized_datasets = datasets.map(lambda x: prepare_train_features(x, tokenizer, pad_on_right), 
                                    batched=True, remove_columns=datasets["train"].column_names)

    # The original Q&A model
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)
    if not args.pretrain:
        # randomize all the layers of the model
        for name, param in model.distilbert.transformer.named_modules():
            if isinstance(param, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                model._init_weights(param)
                print(f"Randomizing the weights of layer {name}")

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

    data_collator = default_data_collator

    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    preprocess_layers(args, model, model_type=args.model_checkpoint)

    # Save model for later evaluation
    trainer.save_model(os.path.join(args.ckpt_folder, 'checkpoint-0'))
    
    # Main train function
    trainer.train()
    wandb.finish()

    if args.eval:

        wandb.init(name = args.ckpt_folder + '_eval')

        final_evals = {}
        steps = np.arange(0, args.max_steps+1, args.save_steps)
        for step in steps:

            ckpt_folder_step = os.path.join(args.ckpt_folder, f"checkpoint-{step}")
            model = AutoModelForQuestionAnswering.from_pretrained(ckpt_folder_step)

            final_eval = eval_f1(tokenized_datasets, tokenizer, model, datasets, pad_on_right)
            final_evals[step] = final_eval
            print(f"Step {step}: exact match score is {final_eval['exact_match']}," + 
                    f" and f1 is {final_eval['f1']}")
            wandb.log({"EM": final_eval['exact_match'], "f1": final_eval['f1']})

            if args.eval_ww:
                watcher = ww.WeightWatcher(model=model)
                details = watcher.analyze()
                summary = watcher.get_summary(details)
                wandb.log({"alpha": summary['alpha']})

                ww_results = {"details": details, "summary": summary}
                pickle.dump(ww_results, open( os.path.join(ckpt_folder_step, "ww_results.pkl"), "wb" ) )

        pickle.dump(final_evals, open( os.path.join(args.ckpt_folder, "f1.pkl"), "wb" ) )
        

if __name__ == "__main__":

    print("Experiment starting. Transformer version is:")
    print(transformers.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder", type=str, default="tmp_SQuAD")
    parser.add_argument("--model_checkpoint", type=str, default="distilbert-base-uncased")
    parser.add_argument("--squad_v2", action='store_true', help="use squad v2")
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
    parser.add_argument("--eval", action='store_true', help="evaluate some metrics after training")
    parser.add_argument("--eval_ww", action='store_true', help="evaluate weightwatcher after training")
    parser.add_argument("--max_length", type=int, default=384, 
                        help="The maximum length of a feature (question and context)")
    parser.add_argument("--doc_stride", type=int, default=128,
                help="The authorized overlap between two part of the context when splitting it is needed.")

    args = parser.parse_args()

    main()