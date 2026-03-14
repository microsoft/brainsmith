#!/usr/bin/env python3
"""
Apply PTQ Quantization using Brevitas to FP32 Model and Export to Clean ONNX
"""

import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import argparse
import os
import numpy as np
from tqdm import tqdm
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.general import (
    RemoveUnusedTensors,
    SortGraph,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)

from utils import create_tinybert_config, fetch_dataloader, validate_model, train_model
from quant_utils import apply_calibration, apply_qronos, apply_bert_quantization

def load_fp32_model(model_path, max_length=128):
    """Load the trained FP32 model"""
    print(f"Loading FP32 model from {model_path}...")
    config = create_tinybert_config()
    model = BertForSequenceClassificationWrapper(config, max_length)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    return model


class BertForSequenceClassificationWrapper(BertForSequenceClassification):
    def __init__(self, config, max_length=128):
        super().__init__(config)
        self.max_length = max_length

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        attention_mask = torch.ones((batch_size, self.max_length), dtype=torch.long, device=input_ids.device)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask)


def apply_qonnx_cleanup(model_path):
    """Apply QONNX cleanup transformations to reduce complexity"""

    try:
        model = ModelWrapper(model_path)

        print(f"  Original model has {len(model.graph.node)} nodes")

        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveUniqueParameterTensors())
        model = model.transform(SortGraph())
        model = model.transform(FoldConstants())
        model = model.transform(RemoveUnusedTensors())

        model = model.transform(FoldTransposeIntoQuantInit())

        print(f"  Cleaned model has {len(model.graph.node)} nodes")

        cleaned_path = model_path.replace('.onnx', '_cleaned.onnx')
        model.save(cleaned_path)

        print(f"  Cleaned model saved to: {cleaned_path}")
        return cleaned_path

    except Exception as e:
        print(f"  QONNX cleanup failed: {e}")
        return model_path


def export_quantized_to_onnx(model, output_path, max_length=128):
    """Export quantized model to clean ONNX"""
    device = next(model.parameters()).device
    model.eval()

    dummy_input = torch.ones(1, max_length, dtype=torch.long).to(device)

    from brevitas.export import export_qonnx
    print(f"Attempting QONNX export with dynamo=True...")
    export_qonnx(model, dummy_input, output_path, dynamo=True)
    print(f"QONNX export successful")

    print(f"Quantized ONNX model saved to: {output_path}")
    cleaned_path = apply_qonnx_cleanup(output_path)

    return cleaned_path

def main():
    parser = argparse.ArgumentParser(description='Quantize FP32 Model to INT8 and Export to ONNX')
    parser.add_argument('--input_model', default='best_fp32_model.pth',
                        help='Path to FP32 PyTorch model (default: %(default)s)')
    parser.add_argument('--output', default='quantized_int8_model.onnx',
                        help='Output quantized ONNX path (default: %(default)s)')
    parser.add_argument('--calibration_samples', type=int, default=512,
                        help='Number of samples for calibration (default: %(default)s)')
    parser.add_argument('--bitwidth', type=int, default=8,
                        help='Quantization bit width (default: %(default)s)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length (default: %(default)s)')
    parser.add_argument('--qronos', action='store_true',
                        help='Apply Qronos as PTQ (default: %(default)s)')
    parser.add_argument('--qat', action='store_true',
                        help='Apply 1 epoch of quantization-aware training (default: %(default)s)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate & test float/quantized model accuracy (default: %(default)s)')
    parser.add_argument('--max_num_samples', type=int, default=None, help='Crop the training / validation set to a maximum number of samples. None=no cropping. (default: %(default)s)')

    args = parser.parse_args()

    if not os.path.exists(args.input_model):
        print(f"Error: Input model not found at {args.input_model}")
        print("Please run train_fp32_model.py first")
        return

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    calibration_loader = fetch_dataloader(tokenizer, args.calibration_samples, args.max_length, split="train")
    validation_loader = fetch_dataloader(tokenizer, args.max_num_samples, args.max_length, split="validation")
    original_model = load_fp32_model(args.input_model, args.max_length)
    original_model.to(device)

    if args.validate:
        print("Validating float model...")
        float_acc, _ = validate_model(original_model, validation_loader)
        print(f"Float model accuracy: {float_acc:.2f}%")

    config = create_tinybert_config()
    quantized_model = apply_bert_quantization(original_model, config, args.bitwidth, args.max_length)
    quantized_model.to(device)

    print(f"Quantized model has {sum(p.numel() for p in quantized_model.parameters()):,} parameters")

    apply_calibration(quantized_model, calibration_loader)

    if args.validate:
        print("Validating quant model after calibration...")
        quant_acc, _ = validate_model(quantized_model, validation_loader)
        print(f"Quant model accuracy: {quant_acc:.2f}%")

    if args.qronos:
        apply_qronos(
            quantized_model,
            calibration_loader,
            alpha=1e-4,
            act_order=True,
            block_name=None)

        if args.validate:
            print("Validating quant model after Qronos...")
            quant_acc, _ = validate_model(quantized_model, validation_loader)
            print(f"Quant model accuracy: {quant_acc:.2f}%")

    if args.qat:
        train_loader = fetch_dataloader(tokenizer, args.max_num_samples, args.max_length, split="train")
        train_model(quantized_model, train_loader, validation_loader, device, epochs=1)

    cleaned_model_path = export_quantized_to_onnx(quantized_model, args.output, args.max_length)

    torch.save(quantized_model.state_dict(), 'quantized_int8_model.pth')

    print(f"\nQuantization completed!")
    print(f"Quantized ONNX model saved to: {args.output}")
    if cleaned_model_path != args.output:
        print(f"Cleaned ONNX model saved to: {cleaned_model_path}")
    print(f"Quantized PyTorch model saved to: quantized_int8_model.pth")


if __name__ == "__main__":
    main()
