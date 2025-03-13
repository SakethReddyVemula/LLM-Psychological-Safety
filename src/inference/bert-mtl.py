import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, jaccard_score, hamming_loss
import numpy as np
from tqdm import tqdm

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Emotion labels
EMOTION_LABELS = ["anger", "brain dysfunction", "emptiness", "hopelessness",
                  "loneliness", "sadness", "suicide intent", "worthlessness"]

# Load DepressionEmo dataset (multi-label emotion classification)
def load_depressionemo(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [entry["post"] for entry in data]
    labels = [entry["emotions"] for entry in data]
    
    # Convert multi-labels to binary format
    mlb = MultiLabelBinarizer(classes=EMOTION_LABELS)
    label_matrix = mlb.fit_transform(labels)
    
    return texts, label_matrix

# Load Identifying-Depression dataset (binary classification)
def load_identifying_depression(depression_folder, non_depression_folder):
    texts, labels = [], []
    
    # Depressed blogs
    for filename in os.listdir(depression_folder):
        with open(os.path.join(depression_folder, filename), "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
            labels.append(1)  # Depressed label
    
    # Non-depressed blogs
    for filename in os.listdir(non_depression_folder):
        with open(os.path.join(non_depression_folder, filename), "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
            labels.append(0)  # Non-depressed label
    
    return texts, labels

# Load datasets
emo_texts, emo_labels = load_depressionemo("train_fixed.json")
dep_texts, dep_labels = load_identifying_depression("blogs_depression", "blogs_non_depression")

# Split into train and test sets
emo_train_texts, emo_test_texts, emo_train_labels, emo_test_labels = train_test_split(emo_texts, emo_labels, test_size=0.2, random_state=42)
dep_train_texts, dep_test_texts, dep_train_labels, dep_test_labels = train_test_split(dep_texts, dep_labels, test_size=0.2, random_state=42)

# Tokenization function
# def tokenize_texts(texts, max_length=512):
#     return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def tokenize_texts(texts, max_length=512):
    return tokenizer(
        texts,
        padding="longest",  # Ensure consistent padding within the batch
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Custom Dataset Class
class MultiTaskDataset(Dataset):
    def __init__(self, texts, emo_labels=None, dep_labels=None):
        self.texts = texts
        self.emo_labels = emo_labels
        self.dep_labels = dep_labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = tokenize_texts([text])
        
        item = {key: val.squeeze(0) for key, val in inputs.items()}  # Flatten tensors
        
        # Ensure labels are not None before converting to tensor
        if self.emo_labels is not None and self.emo_labels[idx] is not None:
            item["emo_labels"] = torch.tensor(self.emo_labels[idx], dtype=torch.float)
        else:
            item["emo_labels"] = None  # Keep it None for safety

        if self.dep_labels is not None and self.dep_labels[idx] is not None:
            item["dep_labels"] = torch.tensor(self.dep_labels[idx], dtype=torch.float)
        else:
            item["dep_labels"] = None  # Keep it None for safety
        
        return item

# Create DataLoaders
batch_size = 16

train_dataset = MultiTaskDataset(emo_train_texts + dep_train_texts, 
                                 emo_train_labels.tolist() + [[0]*8]*len(dep_train_texts), 
                                 [None]*len(emo_train_texts) + dep_train_labels)

test_dataset = MultiTaskDataset(emo_test_texts + dep_test_texts, 
                                emo_test_labels.tolist() + [[0]*8]*len(dep_test_texts), 
                                [None]*len(emo_test_texts) + dep_test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define Multi-Task Model
class MultiTaskBERT(nn.Module):
    def __init__(self, bert_model="bert-base-uncased"):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        
        # Task-specific classification heads
        self.emo_head = nn.Linear(self.bert.config.hidden_size, 8)  # 8 emotions
        self.dep_head = nn.Linear(self.bert.config.hidden_size, 1)  # Binary classification
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Get CLS token representation
        
        emo_logits = self.emo_head(pooled_output)
        dep_logits = self.dep_head(pooled_output).squeeze(-1)  # Squeeze to get scalar output
        
        return emo_logits, dep_logits

# Multi-GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskBERT()
model = nn.DataParallel(model, device_ids=[0, 1, 2])  # Use 3 GPUs
model = model.to(device)

# Optimizer & Loss Functions
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion_emo = nn.BCEWithLogitsLoss()
criterion_dep = nn.BCEWithLogitsLoss()

# Training function
def train_model(model, train_loader, optimizer, epochs=3):
    model.train()
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        for batch in tqdm(train_loader, desc="Batch processing"):
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            
            emo_labels = batch.get("emo_labels")
            dep_labels = batch.get("dep_labels")

            optimizer.zero_grad()
            emo_logits, dep_logits = model(input_ids, attention_mask)

            loss = 0
            if emo_labels is not None:
                emo_labels = emo_labels.to(device)
                loss += criterion_emo(emo_logits, emo_labels)

            if dep_labels is not None:
                dep_labels = dep_labels.to(device)
                loss += criterion_dep(dep_logits, dep_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def evaluate(model, test_loader):
    model.eval()
    all_emo_preds, all_emo_labels = [], []
    all_dep_preds, all_dep_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            emo_labels = batch.get("emo_labels")
            dep_labels = batch.get("dep_labels")

            # Forward pass through model
            emo_logits, dep_logits = model(input_ids, attention_mask)

            # Convert logits to probabilities
            emo_preds = torch.sigmoid(emo_logits).detach().cpu().numpy()
            dep_preds = torch.sigmoid(dep_logits).detach().cpu().numpy()

            # Store multi-label emotion predictions
            if emo_labels is not None:
                all_emo_preds.append(emo_preds)
                all_emo_labels.append(emo_labels.to(device).detach().cpu().numpy())

            # Store binary depression predictions
            if dep_labels is not None:
                all_dep_preds.append((dep_preds > 0.5).astype(int))  # Convert probabilities to binary
                all_dep_labels.append(dep_labels.to(device).detach().cpu().numpy())

    # Convert lists to numpy arrays
    all_emo_preds = np.vstack(all_emo_preds)
    all_emo_labels = np.vstack(all_emo_labels)
    all_dep_preds = np.concatenate(all_dep_preds).astype(int)
    all_dep_labels = np.concatenate(all_dep_labels).astype(int)

    # ---- Multi-Label Emotion Classification Metrics ----
    print("\n---- Multi-Label Emotion Classification ----")
    print(f"Macro F1-score: {f1_score(all_emo_labels, all_emo_preds > 0.5, average='macro'):.4f}")
    print(f"Micro F1-score: {f1_score(all_emo_labels, all_emo_preds > 0.5, average='micro'):.4f}")
    print(f"Weighted F1-score: {f1_score(all_emo_labels, all_emo_preds > 0.5, average='weighted'):.4f}")
    print(f"Precision (macro): {precision_score(all_emo_labels, all_emo_preds > 0.5, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(all_emo_labels, all_emo_preds > 0.5, average='macro'):.4f}")
    print(f"Hamming Loss: {hamming_loss(all_emo_labels, all_emo_preds > 0.5):.4f}")
    print(f"Jaccard Score (macro): {jaccard_score(all_emo_labels, all_emo_preds > 0.5, average='macro'):.4f}")

    # ---- Binary Depression Classification Metrics ----
    print("\n---- Binary Depression Classification ----")
    print(f"Accuracy: {accuracy_score(all_dep_labels, all_dep_preds):.4f}")
    print(f"Precision: {precision_score(all_dep_labels, all_dep_preds):.4f}")
    print(f"Recall: {recall_score(all_dep_labels, all_dep_preds):.4f}")
    print(f"F1-score: {f1_score(all_dep_labels, all_dep_preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(all_dep_labels, all_dep_preds):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_dep_labels, all_dep_preds)
    print("\nConfusion Matrix:")
    print(cm)

# Call evaluation on multi-GPU model
evaluate(model, test_loader)
