from tqdm import tqdm

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertConfig

def fetch_dataloader(tokenizer, num_samples=None, max_length=128, split="train", seed=42, batch_size=32):
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    dataset = load_dataset("glue", "sst2")
    samples = dataset[split].shuffle(seed=seed)
    if num_samples is not None:
        samples = samples.select(range(num_samples))
    data = samples.map(tokenize_function, batched=True)
    data.set_format(type="torch", columns=["input_ids", "label"])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=split == "train")
    return dataloader


def get_logits_from_outputs(outputs):
    # Handle different output formats
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, dict) and 'logits' in outputs:
        return outputs['logits']
    else:
        # If it's a tensor or other format, assume it's the logits directly
        return outputs


def validate_model(model, validation_loader, criterion=None):

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_loader, desc="Validating")):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)
            logits = get_logits_from_outputs(outputs)
            if criterion is not None:
                batch_loss = criterion(logits, labels)
                loss += batch_loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).to(dtype=torch.int64).sum().item()

    acc = correct / len(validation_loader.dataset) * 100

    return acc, loss

def train_model(model, train_loader, val_loader, device, epochs=3, output_file=None):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            logits = get_logits_from_outputs(outputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_acc, val_loss = validate_model(model, val_loader, criterion=criterion)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if output_file is not None:
                torch.save(model.state_dict(), output_file)
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    return best_val_acc

def create_tinybert_config():
    """Create TinyBERT configuration"""
    config = BertConfig(
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=1536,
        hidden_act="relu",
        num_labels=2,
    )
    return config
