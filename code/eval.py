import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

models = {
#     "suicide_ideation": "results/suicide_ideation/checkpoint-1500",
#     "implicit_toxicity": "results/implicit_toxicity/checkpoint-1500"
# "depression" : "results/depression/checkpoint-300" 
}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Tokenization helper
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

for task, model_path in models.items():
    print(f"\nEvaluating {task}...")

    # Load model
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)

    # Load and prepare data
    df = pd.read_csv(f"{task}_dataset.csv")
    print(f"Samples: {len(df)}")
    
    df["text"] = f"[{task}] " + df["text"]
    df["label"] = df["label"].astype(int)

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    # Dummy training arguments (required by Trainer)
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(eval_dataset=tokenized_dataset)
    print(results)
