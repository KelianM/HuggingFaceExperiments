import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import ClassLabel
import random
import pandas as pd

BASE_MODEL_NAME = "distilbert/distilgpt2"
MODEL_SAVE_PATH = "./checkpoints/fine_tuned_distilgpt2"
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
NUM_WARM_UP_STEPS = 0

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    print(df)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def train_model(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, device, num_epochs):
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        metric = evaluate.load("accuracy")
        
        model.train()
        print(f"Epoch {epoch + 1} of {num_epochs}...")
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        print("Validating...")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).flatten()
                metric.add_batch(predictions=predictions, references=batch["labels"].flatten())
            print("Validation metrics:", metric.compute())
    return model

# Load wikitext dataset for finetuning
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
show_random_elements(datasets["train"])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Format dataset for pytorch
tokenized_datasets = datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Get smaller dataset for finetuning
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

# Create dataset loaders with a LM collator for self-supervision (labels are the input shifted by one token)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(small_train_dataset, collate_fn=collator, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(small_eval_dataset, collate_fn=collator, batch_size=BATCH_SIZE)

# Load pretrained model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype="auto")
model.config.loss_type = "ForCausalLMLoss"

# Finetune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=NUM_WARM_UP_STEPS, num_training_steps=num_training_steps
)
finetuned_model = train_model(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, device, NUM_EPOCHS)
finetuned_model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

# Chat with model using `transformers-cli chat --model_name_or_path MODEL_SAVE_PATH`