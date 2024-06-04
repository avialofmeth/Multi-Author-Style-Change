# !pip install torch torchvision torchaudio wandb transformers datasets  torchmetrics pip install transformers[torch]

import os
import torch
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
import torchmetrics
import wandb

import json
import numpy as np
import time
from datasets import Dataset


from utilities import (read_paragraphs,
                       read_ground_truth, 
                       generate_dataset)
# Open the JSON file
with open('./config/fine_tune_bert_base.json', 'r') as f:
    fine_tune_conf = json.load(f)

# 更新路径 Path prefixes
train_directory = './data/train_processed'
train_label_directory = './data/train_label'

# Due to the lack of the true test set. We use the validation set as our test set.
# We will split the training set into train and validation sets.
test_directory = './data/validation_processed'
test_label_directory = './data/validation_label'

checkpoint = fine_tune_conf['checkpoint']
run_name = fine_tune_conf["run_name_prefix"] + checkpoint
# 读取段落数据
# Read documents
# max(end_id) = 4200
train_data = read_paragraphs(train_directory, 
                             start_id=fine_tune_conf['train_data_start_id'], 
                             end_id=fine_tune_conf['train_data_end_id']) # {'problem-x': [sen 1, sen 2, ...], ...}
# max(end_id) = 900
test_data = read_paragraphs(test_directory, 
                            start_id=fine_tune_conf['test_data_start_id'], 
                            end_id=fine_tune_conf['test_data_end_id'])
# 读取 ground truth 数据
# Read ground truth labels
train_labels = read_ground_truth(train_label_directory, 
                                 start_id=fine_tune_conf['train_data_start_id'], 
                                 end_id=fine_tune_conf['train_data_end_id']) # {'problem-x': [1, ...], ...}
test_labels  = read_ground_truth(test_label_directory, 
                                 start_id=fine_tune_conf['test_data_start_id'], 
                                 end_id=fine_tune_conf['test_data_end_id'])

tokenizer = BertTokenizer.from_pretrained(checkpoint)

train_dataset = generate_dataset(train_data, train_labels, tokenizer)
test_dataset = generate_dataset(test_data, test_labels, tokenizer)

training_sets = train_dataset.train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
training_sets["validation"] = training_sets.pop("test")
# Add the "test" set to our `DatasetDict`
training_sets["test"] = test_dataset

# training_sets 

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], 
                     truncation=True)

tokenized_datasets = training_sets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# tokenized_datasets


wandb.login()
# wandb.init(project="Multi_author_test") 


my_metrics = {"F1": torchmetrics.classification.F1Score(task='binary', num_classes=2, average="macro"), 
            'Accuracy': torchmetrics.classification.BinaryAccuracy()}

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    labels = torch.from_numpy(labels)
    predictions = torch.from_numpy(np.argmax(logits, axis=-1))
    eval_result = {}
    for key, me in my_metrics.items():
        eval_result[key] = me(predictions, labels).item()
    return eval_result

# hyper-parameter search, using sweep
sweep_config = {
                'method': 'random',
                'metric': {'goal': 'maximize', 'name': 'eval/F1'},
                'parameters': {
                    'batch_size': {
                        'distribution': 'q_log_uniform_values',
                        'max': fine_tune_conf['sweep_config']['max_batch_size'],
                        'min': fine_tune_conf['sweep_config']['min_batch_size']
                    },
                    'epochs': {'values': fine_tune_conf['sweep_config']['epochs']},
                    'learning_rate': {'distribution': 'uniform',
                                      'max': fine_tune_conf['sweep_config']['max_learning_rate'],
                                      'min': fine_tune_conf['sweep_config']['min_learning_rate']},
                    'optimizer': {'values': ['adam']}
                }
 }

sweep_id = wandb.sweep(sweep_config, project=fine_tune_conf["sweep_project_name"])

def train(config=None):
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # On the full dataset: with batch size as 24 and gradient_accumulation_steps=4, 
    # there will be 546 training steps and 182 validation steps
    training_args = TrainingArguments(
        output_dir=f"finetuned-{checkpoint}",
        evaluation_strategy = "steps",
        eval_steps=fine_tune_conf['training_arguments']['eval_steps'],
        gradient_accumulation_steps=4,
        save_strategy='epoch',
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size, 
        per_device_eval_batch_size=16,
        num_train_epochs=config.epochs,
        weight_decay=0.01,
        report_to="wandb",  # enable logging to W&B
        logging_steps=2,  # how often to log to W&B
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate(tokenized_datasets['validation'])

    # trainer.evaluate(tokenized_datasets['test'])
    torch.cuda.empty_cache()

wandb.agent(sweep_id, train, count=fine_tune_conf['training_arguments']['sweep_count'])
# wandb.finish()


