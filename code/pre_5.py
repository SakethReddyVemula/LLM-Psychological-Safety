import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer once
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def process_datasets(task_name, config, target_samples=10000):
    # Read CSV
    df = pd.read_csv(config["file"])

    # Add task token to the text
    df["text"] = f"[{task_name}] " + df["text"]
    df["label"] = df["label"].astype(int)

    # Batch tokenize texts to compute token lengths without truncation
    tokenized = tokenizer(df["text"].tolist(), truncation=False, add_special_tokens=True)
    # Compute token lengths for each example
    df["token_length"] = [len(ids) for ids in tokenized["input_ids"]]

    # Filter out examples with >512 tokens
    filtered_df = df[df["token_length"] <= 512].drop(columns=["token_length"])
    
    # Ensure the dataset has exactly target_samples examples if possible
    if len(filtered_df) > target_samples:
        filtered_df = filtered_df.sample(n=target_samples, random_state=42).reset_index(drop=True)
    else:
        filtered_df = filtered_df.reset_index(drop=True)

    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(filtered_df)

    # Split into train/test (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Tokenize dataset with truncation and padding (batch processing)
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), 
        batched=True
    )
    # Remove columns that are not needed for training
    keep_cols = ["input_ids", "attention_mask", "label"]
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in keep_cols])
    dataset.set_format("torch")
    
    print(f"✅ Processed {config['file']} for task '{task_name}' with {len(filtered_df)} examples.")
    print(f"✅ Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}") 
    
    return dataset

# Define metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# # Start with a base model
# model = None

# # Loop through tasks
task_configs = {
    # "sentiment": {"file": "sentiment_dataset.csv", "num_labels": 3},
    # "emotion": {"file": "emotion_dataset.csv", "num_labels": 6},
    # "depression": {"file": "depression_dataset.csv", "num_labels": 2},
    "suicide_ideation": {"file": "suicide_dataset.csv", "num_labels": 2},
    "implicit_toxicity": {"file": "toxicity_dataset.csv", "num_labels": 2}
}

for task_name, config in task_configs.items():
    print(f"\n🔧 Fine-tuning on task: {task_name}")

#     # Load and preprocess dataset (ensuring 10k samples after filtering if available)
    dataset = process_datasets(task_name, config, target_samples=10000)
    
    
#     # Load or reconfigure model for the current task
#     if model is None:
#         model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=config["num_labels"])
#     else:
#         # Reconfigure classifier head for new task
#         model.classifier = torch.nn.Linear(model.config.hidden_size, config["num_labels"])
#         model.num_labels = config["num_labels"]

#     model.to(device)  # Move model to GPU

#     # Define training arguments
#     training_args = TrainingArguments(
#         output_dir=f"./results/{task_name}",
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         logging_dir=f"./logs/{task_name}",
#         logging_steps=10,
#         load_best_model_at_end=True
#     )

#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["test"],
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics
#     )

#     # Train and evaluate
#     trainer.train()
#     metrics = trainer.evaluate()
#     print(f"📊 Metrics for {task_name}: {metrics}")



