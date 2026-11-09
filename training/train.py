import os
import json
import csv
import yaml
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

with open("/kaggle/working/training/config.yaml") as f:
    config = yaml.safe_load(f)

torch.manual_seed(config["seed"])

class CommandDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.data = []
        for r in records:
            prompt = f"Command: {r['text']}\nJSON: "
            target = json.dumps({
                "intent": r["intent"],
                "date": r["date"],
                "time": r["time"],
                "place": r["place"]
            })
            full = prompt + target + tokenizer.eos_token
            encoded = tokenizer(full, max_length=256, truncation=True, padding="max_length")
            prompt_ids = tokenizer(prompt, max_length=256, truncation=True)["input_ids"]
            labels = encoded["input_ids"].copy()
            labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
            labels = [t if m else -100 for t, m in zip(labels, encoded["attention_mask"])]
            self.data.append({
                "input_ids": torch.tensor(encoded["input_ids"]),
                "attention_mask": torch.tensor(encoded["attention_mask"]),
                "labels": torch.tensor(labels)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

with open(config["train_file"], encoding="utf-8") as f:
    records = [json.loads(line) for line in f][:1500]

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(config["model_name"])
model.config.pad_token_id = tokenizer.pad_token_id

dataset = CommandDataset(records, tokenizer)

args = TrainingArguments(
    output_dir="/kaggle/working/models/checkpoints",
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=8,
    num_train_epochs=1,
    seed=config["seed"],
    use_cpu=False,
    fp16=False,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=default_data_collator
)

trainer.train()

model_dir = "/kaggle/working/models/v1"
os.makedirs(model_dir, exist_ok=True)
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

with open("/kaggle/working/training/loss.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss"])
    for item in trainer.state.log_history:
        if "loss" in item:
            writer.writerow([item.get("step"), item["loss"]])

print("TRAINING COMPLETED")
print("Model saved:", model_dir)
