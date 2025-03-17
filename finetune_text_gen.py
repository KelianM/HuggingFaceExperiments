import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

BASE_MODEL_NAME = "google-bert/bert-base-cased"
MODEL_SAVE_PATH = "./checkpoints/fine_tuned_bert"
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_WARM_UP_STEPS = 0

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, device, num_epochs):
    for epoch in range(num_epochs):
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

        model.eval()
        with torch.no_grad():
            all_logits = []
            all_labels = []
            for batch in tqdm(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                all_logits.append(logits)
                all_labels.append(batch["labels"])
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = compute_metrics(all_logits, all_labels)
            print("Validation metrics:", metrics)
    return model

dataset = load_dataset("yelp_review_full")
print(dataset["train"][100])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Format dataset for pytorch
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Get smaller dataset for finetuning
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Create dataset loaders
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)

# Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=5, torch_dtype="auto")

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
metric = evaluate.load("accuracy")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=NUM_WARM_UP_STEPS, num_training_steps=num_training_steps
)
model = train_model(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, device, NUM_EPOCHS)
model.save_pretrained(MODEL_SAVE_PATH)

# Chat with model using `transformers-cli chat --model_name_or_path my_model`