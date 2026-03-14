############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch

# Import brainsmith early to set up paths
import brainsmith
from brainsmith.settings import get_config
# Note: Config export to environment (FINN_ROOT, etc.) happens automatically

from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from onnx.onnx_pb import StringStringEntryProto
from onnxsim import simplify
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup
from torch import nn
from transformers import BertConfig, BertModel
from transformers.utils.fx import symbolic_trace
import brevitas.nn as qnn
import brevitas.onnx as bo

# Import local custom steps to register them for use in blueprint YAML.
# These steps are referenced in bert_demo.yaml: remove_head, remove_tail, generate_reference_io
import custom_steps

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from brainsmith import explore_design_space
from brainsmith.dse.types import SegmentStatus

warnings.simplefilter("ignore")


def generate_bert_model(args):
    """Generate quantized BERT model from HuggingFace with Brevitas quantization.

    This matches the functionality from old end2end_bert.py::gen_initial_bert_model()
    """

    # Global consts used by Brevitas build step
    dtype = torch.float32

    # Create BERT configuration
    config = BertConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        attn_implementation="sdpa",
        hidden_act="relu",
    )
    # Initialize model
    model = BertModel(config=config)
    model.to(dtype=dtype)
    model.eval()
    # Prepare inputs
    vocab_size = model.config.vocab_size
    seq_len = args.seqlen
    batch_size = 1

    input_ids = torch.randint(vocab_size, (batch_size, seq_len), dtype=torch.int64)
    inp = {'input_ids': input_ids}

    # Symbolic tracing
    input_names = inp.keys()
    model = symbolic_trace(model, input_names)

    # Replace SDPA with quantizable layers
    model = replace_sdpa_with_quantizable_layers(model)

    # Configure quantization
    unsigned_hidden_act = config.hidden_act == 'relu'
    layerwise_compute_layer_map = {}

    # Linear layer quantization
    layerwise_compute_layer_map[nn.Linear] = (
        qnn.QuantLinear,
        {
            'input_quant': lambda module: Uint8ActPerTensorFloat
                if module.in_features == config.intermediate_size and unsigned_hidden_act
                else Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'weight_bit_width': args.bitwidth,
            'output_quant': None,
            'bias_quant': None,
            'return_quant_tensor': False
        }
    )
    # Attention quantization
    layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
        qnn.QuantScaledDotProductAttention,
        {
            'softmax_input_quant': Int8ActPerTensorFloat,
            'softmax_input_bit_width': args.bitwidth,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'attn_output_weights_bit_width': args.bitwidth,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'q_scaled_bit_width': args.bitwidth,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'k_transposed_bit_width': args.bitwidth,
            'v_quant': Int8ActPerTensorFloat,
            'v_bit_width': args.bitwidth,
            'attn_output_quant': None,
            'return_quant_tensor': False
        }
    )
    # Tanh quantization
    layerwise_compute_layer_map[nn.Tanh] = (
        qnn.QuantTanh,
        {
            'input_quant': None,
            'act_quant': Int8ActPerTensorFloat,
            'act_bit_width': args.bitwidth,
            'return_quant_tensor': False
        }
    )

    # Apply quantization
    quant_model = layerwise_quantize(model, compute_layer_map=layerwise_compute_layer_map)
    quant_model.to(dtype=dtype)

    # Calibration
    with torch.no_grad(), calibration_mode(quant_model):
        quant_model(**inp)

    # Export to ONNX
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        tmp_path = tmp.name

    with torch.no_grad():
        bo.export_qonnx(
            quant_model,
            (input_ids),
            tmp_path,
            do_constant_folding=True,
            input_names=['input_ids'],
            opset_version=18,
            dynamo=True,
            optimize=True
        )

    # Load and return model
    model = onnx.load(tmp_path)
    os.unlink(tmp_path)

    # Save initial Brevitas model for debugging
    debug_path = os.path.join(args.output_dir, "debug_models")
    os.makedirs(debug_path, exist_ok=True)
    onnx.save(model, os.path.join(debug_path, "00_initial_brevitas.onnx"))
    print(f"  - Model inputs: {len(model.graph.input)} tensors")
    print(f"  - Model outputs: {len(model.graph.output)} tensors")
    print(f"  - Number of nodes: {len(model.graph.node)}")
    return model


def run_brainsmith_dse(model, args):
    """Run Brainsmith with new execution tree architecture."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "intermediate_models")
    os.makedirs(model_dir, exist_ok=True)

    # Extract metadata from the original model
    metadata = {}
    for node in model.graph.node:
        md = {}
        for prop in node.metadata_props:
            md[prop.key] = prop.value
        metadata[node.name] = md

    # Simplify model (matches old hw_compiler.py)
    simp_model_no_md, check = simplify(model)
    if not check:
        raise RuntimeError("Unable to simplify the Brevitas BERT model")

    # Add the metadata back to the simplified model
    simp_model_with_md = simp_model_no_md
    for node in simp_model_no_md.graph.node:
        if node.name in metadata:
            md_props = metadata[node.name]
            for key,value in md_props.items():
                new_md = StringStringEntryProto(key=key,value=value)
                node.metadata_props.append(new_md)

    model = simp_model_with_md
    # Save simplified model
    onnx.save(model, os.path.join(model_dir, "simp.onnx"))
    # Also save to debug directory for comparison
    debug_dir = os.path.join(args.output_dir, "debug_models")
    onnx.save(model, os.path.join(debug_dir, "01_after_simplify.onnx"))

    # Run cleanup
    cleanup(
        in_file=os.path.join(model_dir, "simp.onnx"),
        out_file=os.path.join(args.output_dir, "df_input.onnx")
    )

    # Clean up temporary artifacts (simp.onnx is already saved to debug_models)
    os.remove(os.path.join(model_dir, "simp.onnx"))
    shutil.rmtree(model_dir)

    # Save a copy of the cleaned model for visualization
    debug_dir = os.path.join(args.output_dir, "debug_models")
    os.makedirs(debug_dir, exist_ok=True)
    shutil.copy(
        os.path.join(args.output_dir, "df_input.onnx"),
        os.path.join(debug_dir, "02_after_qonnx_cleanup.onnx")
    )

    # Get blueprint path from args
    blueprint_path = Path(__file__).parent / args.blueprint

    # Create the FPGA accelerator
    results = explore_design_space(
        model_path=os.path.join(args.output_dir, "df_input.onnx"),
        blueprint_path=str(blueprint_path),
        output_dir=args.output_dir
    )

    # Results are automatically logged by explore_design_space()
    # Just check if we succeeded
    stats = results.compute_stats()
    if stats['successful'] == 0:
        raise RuntimeError(f"No successful builds")

    # The new execution tree handles output automatically
    final_model_dst = os.path.join(args.output_dir, "output.onnx")

    # Find the output from the successful execution
    for segment_id, result in results.segment_results.items():
        if result.status == SegmentStatus.COMPLETED and result.output_model:
            shutil.copy2(result.output_model, final_model_dst)
            break
    # Handle shell metadata (matches old hw_compiler.py)
    handover_file = os.path.join(args.output_dir, "stitched_ip", "shell_handover.json")
    if os.path.exists(handover_file):
        with open(handover_file, "r") as fp:
            handover = json.load(fp)
        handover["num_layers"] = args.num_hidden_layers
        with open(handover_file, "w") as fp:
            json.dump(handover, fp, indent=4)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Modern BERT FINN demo - Exact parity with old system using Brainsmith DFC'
    )

    # Model configuration
    parser.add_argument('-o', '--output', help='Output build directory name', required=True)
    parser.add_argument('-z', '--hidden_size', type=int, default=384,
                       help='BERT hidden_size parameter')
    parser.add_argument('-n', '--num_attention_heads', type=int, default=12,
                       help='BERT num_attention_heads parameter')
    parser.add_argument('-l', '--num_hidden_layers', type=int, default=1,
                       help='Number of hidden layers')
    parser.add_argument('-i', '--intermediate_size', type=int, default=1536,
                       help='BERT intermediate_size parameter')
    parser.add_argument('-b', '--bitwidth', type=int, default=8,
                       help='Quantization bitwidth (4 or 8)')
    parser.add_argument('-q', '--seqlen', type=int, default=128,
                       help='Sequence length parameter')

    # Blueprint configuration
    parser.add_argument('--blueprint', type=str, default='bert_demo.yaml',
                       help='Blueprint YAML file to use (default: bert_demo.yaml)')

    # Force flag
    parser.add_argument('--force', action='store_true',
                       help='Remove existing output directory before building')

    args = parser.parse_args()

    # Determine output directory
    build_dir = get_config().build_dir
    args.output_dir = os.path.join(str(build_dir), args.output)

    # Clean up existing directory if --force flag is set
    if args.force and os.path.exists(args.output_dir):
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    print("=" * 60)
    print("BERT Demo - Brainsmith Dataflow Core")
    print("=" * 60)
    print(f"Model: {args.num_hidden_layers} layers, hidden={args.hidden_size}, heads={args.num_attention_heads}, intermediate={args.intermediate_size}")
    print(f"Quantization: {args.bitwidth}-bit, sequence length={args.seqlen}")
    print(f"Blueprint: {args.blueprint}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    try:
        # Step 1: Generate BERT model
        print("\nStep 1: Generating dummy quantized BERT model...")
        model = generate_bert_model(args)

        # Step 2: Create dataflow core accelerator
        print("\nStep 2: Creating dataflow core accelerator...")
        result = run_brainsmith_dse(model, args)

        print("\n" + "=" * 70)
        print("BUILD COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Output directory: {args.output_dir}")
    except Exception as e:
        print(f"\nERROR: Build failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
