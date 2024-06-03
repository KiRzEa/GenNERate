import argparse
import os
import time
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bigscience/bloom-560m")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--data_dir", type=str, default="../data/sentence")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#======================================
model_name = args.model_name
lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
#======================================
print("="*50)
print("[INFO CUDA is Available: ", torch.cuda.is_available())
print("[INFO] Device: ", device)
print("[INFO] Model ID: ", model_name)
print("[INFO] Learning Rate: ", lr)
print("[INFO] Number of Epochs: ", num_epochs)
print("[INFO] Batch Size: ", batch_size)
print("="*50)

peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8, # Lora attention dimension.
        lora_alpha=32, # the alpha parameter for Lora scaling.
        lora_dropout=0.05, # the dropout probability for Lora layers.
        target_modules=[
            "query_key_value"
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
)

dataset = create_dataset(args.data_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model = get_peft_model(base_model, peft_config)
print_trainable_parameters(model)

max_output_length = max([len(tokenizer(review)["input_ids"]) for review in dataset['train']["label"]])
max_input_length = max([len(tokenizer(review)["input_ids"]) for review in dataset['train']["text"]])

def preprocess_function(examples):
    batch_size = len(examples["text"])
    inputs = [item + " " for item in examples["text"]]
    targets = examples["label"]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataloader = DataLoader(
    processed_datasets['train'], shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

start_time= time.time() # set the time at which inference started

# training and evaluation
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        #         print(batch)
        #         print(batch["input_ids"].shape)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

stop_time=time.time()
time_training =stop_time - start_time
print("Training time (seconds): ", time_training)

@torch.inference_mode()
def get_prediction(example):
    input_ids = tokenizer(example, max_length=max_input_length, return_tensors="pt", padding="max_length", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_output_length, eos_token_id=tokenizer.eos_token_id)
    preds = outputs[:, max_input_length:].detach().cpu().numpy()
    label = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return label

model.eval()
start_time= time.time() # set the time at which inference started

test_pred = []
dev_pred = []
for text in tqdm(input_test):
    pred = get_prediction(text)
    test_pred.extend(pred)
for text in tqdm(input_dev):
    pred = get_prediction(text)
    dev_pred.append(pred)

stop_time=time.time()
inference_time = stop_time - start_time
print("Inference time (seconds): ", inference_time)

df = pd.DataFrame(list(zip(input_test, output_test, test_pred)),
               columns =['text','gold', 'pred'])
df.to_csv(model_name.replace("/", "_") + ".csv",index=False)