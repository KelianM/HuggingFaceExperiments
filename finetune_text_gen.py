import torch
import evaluate
import json
from collections import deque
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import ClassLabel
import random
import pandas as pd
from peft import LoftQConfig, LoraConfig, get_peft_model

BASE_MODEL_NAME = "google/gemma-3-1b-it"
ATTN_IMPL = 'eager' # Use 'eager' attention implementation as recommended for Gemma3 models
MODEL_SAVE_PATH = "./checkpoints/fine_tuned_gemma"
DATA_PATH = "data/kelian_gpt_hist/conversations.json"
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_WARM_UP_STEPS = 0
# PEFT config
LORA_RANK = 64
LORA_ALPHA = 32
LOFTQ_BITS = 4
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

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

def parse_conversation(mapping):
    """
    Extract a single conversation as a single text block,
    skipping system messages, etc.
    """
    # Get the root node
    root_id = "client-created-root"
    if root_id not in mapping:
        return ""

    text_fragments = []
    queue = deque(mapping[root_id]["children"])  # Start traversal from root's children

    while queue:
        node_id = queue.popleft()
        node = mapping[node_id]
        
        # Add children to queue to continue traversal
        for child_id in node.get("children", []):
            queue.append(child_id)

        msg = node.get("message", {})
        author_role = msg.get("author", {}).get("role")
        # 'parts' hold the actual text
        parts = msg.get("content", {}).get("parts", [])

        # Skip system or invisible messages
        if author_role in ("user", "assistant"):
            # Optionally prepend "User:" or "Assistant:" etc.:
            combined_text = " ".join(parts).strip()
            if combined_text:
                text_fragments.append(f"{author_role.title()}: {combined_text}")

    # Join each turn into one text block (you could also store each turn as separate samples)
    return "\n".join(text_fragments)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

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

            # Free memory
            del batch, outputs, loss
            torch.cuda.empty_cache()

        print("Validating...")
        model.eval()
        metric = evaluate.load("accuracy")  # Reset metric each validation phase

        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).flatten()
                
                metric.add_batch(predictions=predictions, references=batch["labels"].flatten())

                # Free memory
                del batch, outputs, logits, predictions
                torch.cuda.empty_cache()
    return model

# Load the entire file
with open(DATA_PATH, "r") as f:
    all_conversations = json.load(f)

samples = []
for conv in all_conversations:
    text = parse_conversation(conv["mapping"])
    if text.strip():
        samples.append({"text": text})

# Convert to a huggingface Dataset
dataset = Dataset.from_list(samples)
train_size = int(0.9 * len(dataset))
train_dataset = dataset.select(range(train_size))
test_dataset  = dataset.select(range(train_size, len(dataset)))
datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

# Load wikitext dataset for finetuning
show_random_elements(datasets["train"])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
# Use end of sequence token as default padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Format dataset for pytorch
tokenized_datasets = datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Create dataset loaders with a LM collator for self-supervision (labels are the input shifted by one token)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(tokenized_datasets["train"], collate_fn=collator, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=collator, batch_size=BATCH_SIZE)

# Load pretrained model with qunatization for reduced memory usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME,
                                                  torch_dtype="auto",
                                                  attn_implementation=ATTN_IMPL).to(device)
# Initialize PEFT model with LoraConfig and LOFTQ weight initialization
loftq_config = LoftQConfig(loftq_bits=LOFTQ_BITS)
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
    init_lora_weights="loftq",
    loftq_config=loftq_config
)
peft_model = get_peft_model(base_model, lora_config)

# Finetune the model
metric = evaluate.load("accuracy")
optimizer = AdamW(peft_model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=NUM_WARM_UP_STEPS, num_training_steps=num_training_steps
)
finetuned_model = train_model(peft_model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, device, NUM_EPOCHS)
finetuned_model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

# Chat with model using `transformers-cli chat --model_name_or_path MODEL_SAVE_PATH`