import os
import sys
import math

import numpy as np
import tensorflow as tf

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer




# arguments
# python finetune.py username/target_euph models
DATA_PATH = sys.argv[1]
SAVEMODEL_PATH = sys.argv[2]

SEED_VALUE = 666
LABELS_NUM = 2
tf.random.set_seed(SEED_VALUE)
tf.keras.utils.set_random_seed(SEED_VALUE)
tf.config.experimental.enable_op_determinism()


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

PRETRAINED_MODEL="xlm-roberta-large"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=LABELS_NUM)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True)



dataset = load_dataset(DATA_PATH)
dataset = dataset.shuffle(SEED_VALUE)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.class_encode_column("label")
tokenized_datasets = tokenized_datasets.rename_column("label","labels")

tokenized_train, tokenized_test = tokenized_datasets['train'].train_test_split(test_size=0.1).values()

# - - - - - tutorial code - - - - -
# https://raphaelb.org/posts/freezing-bert/

def get_random_seed():
    return int.from_bytes(os.urandom(4), "big")

# define metrics and metrics function
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
auc_metric = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }

args_dict = {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "logging_first_step": True,
        "save_total_limit": 1,
        "fp16": True,
        "dataloader_num_workers": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "seed": get_random_seed(),
    }

freeze_layer_count = 20

if freeze_layer_count:
    # We freeze here the embeddings of the model
    for param in model.roberta.embeddings.parameters():
    #for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    if freeze_layer_count != -1:
	    # if freeze_layer_count == -1, we only freeze the embedding layer
	    # otherwise we freeze the first `freeze_layer_count` encoder layers
        for layer in model.roberta.encoder.layer[:freeze_layer_count]:
        #for layer in model.bert.encoder.layer[:freeze_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False

epoch_steps = len(tokenized_train) / args_dict["per_device_train_batch_size"]
args_dict["warmup_steps"] = math.ceil(epoch_steps)  # 1 epoch
args_dict["logging_steps"] = max(1, math.ceil(epoch_steps))  # 0.5 epoch
args_dict["save_steps"] = args_dict["logging_steps"]

training_args = TrainingArguments(output_dir="test_trainer", **args_dict)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    )

trainer.train()

trainer.save_model(SAVEMODEL_PATH)