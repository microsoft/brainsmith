#!/usr/bin/env python3
"""
Train FP32 TinyBERT Classification Model and Export to Clean ONNX
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import numpy as np
import onnx
import onnxsim
import argparse
import os
from tqdm import tqdm

from utils import create_tinybert_config, fetch_dataloader, validate_model, train_model

def export_to_onnx(model, tokenizer, output_path, max_length=128):
    """Export model to clean ONNX format"""
    print("Exporting to ONNX...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.ones(1, max_length, dtype=torch.long).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    # Simplify ONNX model
    print("Simplifying ONNX model...")
    model_onnx = onnx.load(output_path)
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_onnx, output_path)
    
    print(f"Clean ONNX model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train FP32 TinyBERT and Export to ONNX')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: %(default)s)')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length (default: %(default)s)')
    parser.add_argument('--max_num_samples', type=int, default=None, help='Crop the training / validation set to a maximum number of samples. None=no cropping. (default: %(default)s)')
    parser.add_argument('--output', default='fp32_model.onnx', help='Output ONNX path (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and create model
    print("Loading tokenizer and creating model...")
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    config = create_tinybert_config()
    model = BertForSequenceClassification(config)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load data
    train_loader = fetch_dataloader(tokenizer, num_samples=args.max_num_samples, max_length=args.max_length, split="train", batch_size=args.batch_size)
    val_loader = fetch_dataloader(tokenizer, num_samples=args.max_num_samples, max_length=args.max_length, split="validation", batch_size=args.batch_size)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Train model
    best_acc = train_model(model, train_loader, val_loader, device, args.epochs, output_file='best_fp32_model.pth')
    
    # Load best model for export
    model.load_state_dict(torch.load('best_fp32_model.pth'))
    model.eval()
    
    # Export to ONNX
    export_to_onnx(model, tokenizer, args.output, args.max_length)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"FP32 ONNX model saved to: {args.output}")
    print(f"PyTorch model saved to: best_fp32_model.pth")


if __name__ == "__main__":
    main()
