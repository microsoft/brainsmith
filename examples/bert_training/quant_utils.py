from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.qronos import Qronos
from brevitas.utils.python_utils import recurse_getattr
from brevitas.utils.torch_utils import StopFwdException
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.graph import ModuleToModuleByInstance
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph import TorchFunctionalToModule
from brevitas.nn import ScaledDotProductAttention
import torch.nn.functional as F
from transformers.utils.fx import symbolic_trace


def replace_sdpa_with_quantizable_layers(model):
    """Replace scaled dot product attention with quantizable version"""
    fn_to_module_map = ((F.scaled_dot_product_attention, ScaledDotProductAttention),)
    model = TorchFunctionalToModule(fn_to_module_map=fn_to_module_map).apply(model)
    return model


def apply_bert_quantization(model, config, bitwidth=8, seqlen=128):
    """Apply BERT-style quantization using layerwise approach"""
    print(f"Applying BERT-style quantization with {bitwidth}-bit precision...")

    dtype = torch.float32
    model.to(dtype=dtype)
    model.eval()
    vocab_size = model.config.vocab_size
    batch_size = 1

    input_ids = torch.randint(vocab_size, (batch_size, seqlen), dtype=torch.int64)
    inp = {'input_ids': input_ids}

    print("Performing symbolic tracing...")
    input_names = inp.keys()
    model = symbolic_trace(model, input_names, disable_check=True)

    print("Replacing SDPA with quantizable variants...")
    model = replace_sdpa_with_quantizable_layers(model)
    print("Replacement done.")

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
            'weight_bit_width': bitwidth,
            'output_quant': None,
            'bias_quant': None,
            'return_quant_tensor': False
        }
    )

    layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
        qnn.QuantScaledDotProductAttention,
        {
            'softmax_input_quant': Int8ActPerTensorFloat,
            'softmax_input_bit_width': bitwidth,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'attn_output_weights_bit_width': bitwidth,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'q_scaled_bit_width': bitwidth,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'k_transposed_bit_width': bitwidth,
            'v_quant': Int8ActPerTensorFloat,
            'v_bit_width': bitwidth,
            'out_quant': Int8ActPerTensorFloat,
            'out_bit_width': bitwidth,
            'return_quant_tensor': False
        }
    )

    # HardTanh quantization (replacing Tanh)
    layerwise_compute_layer_map[nn.Tanh] = (
        qnn.QuantHardTanh,
        {
            'input_quant': None,
            'act_quant': Int8ActPerTensorFloat,
            'act_bit_width': bitwidth,
            'min_val': -1.0,
            'max_val': 1.0,
            'return_quant_tensor': False
        }
    )

    print("Applying layerwise quantization...")
    model = layerwise_quantize(
        model=model,
        compute_layer_map=layerwise_compute_layer_map
    )
    model.to(dtype=dtype)

    print("BERT quantization completed.")
    return model


def _dual_optimization_callback(
        model,
        dataloader,
        act_order=True,
        block_name=None,
        group_of_parallel_layers=None,
        algorithm_impl=Qronos):
    """
    This wraps gpfq_mode, which can be used for any layerwise PTQ algorithm that
    optimizes the mismatched objective function || XW - \tilde{X}Q ||, where
    Q is the quantized weights and \tilde{X} are the (potentially quantized)
    activations resulting from the previously quantized layers.

    See https://arxiv.org/abs/2505.11695 for more!
    """
    with gpfq_mode(model,
                   act_order=act_order,
                   group_of_parallel_layers=group_of_parallel_layers,
                   create_weight_orig=True,
                   algorithm_impl=algorithm_impl) as algo:
        algo_model = algo.model
        device = next(algo_model.parameters()).device
        for _ in tqdm(range(algo.num_layers), desc="Applying PTQ (Qronos)"):
            for inps in dataloader:
                input_ids = inps['input_ids'].to(device)
                algo_model(input_ids)
            algo.update()


@torch.no_grad()
def apply_qronos(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        block_name=None,
        alpha=1e-6):
    assert alpha > 0, "Error: alpha needs to be strictly positive"
    # We use the dual optimization callback, which uses two forward passes to correct
    # quantization error in both the weights and activations from previous layers
    print(f"Applying Qronos model with ~{len(dataloader.dataset)} samples...")
    _dual_optimization_callback(
        model,
        dataloader,
        act_order=act_order,
        block_name=block_name,
        group_of_parallel_layers=group_of_parallel_layers,
        algorithm_impl=partial(Qronos, alpha=alpha))
    print("Qronos completed")

def apply_calibration(model, calibration_loader):
    """Calibrate the quantized model with sample data using proper calibration mode"""
    print(f"Calibrating model with ~{len(calibration_loader.dataset)} samples...")

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad(), calibration_mode(model):
        for batch_idx, batch in enumerate(tqdm(calibration_loader, desc="Calibrating")):
            input_ids = batch["input_ids"].to(device)

            _ = model(input_ids)

            if batch_idx >= 50:
                break

    print("Calibration completed")
