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

"""
BERT-Specific Custom Build Steps (Runtime Component Registration Example)

These steps are specific to this BERT example and register when custom_steps.py
is imported by bert_demo.py. They are referenced in bert_demo.yaml and executed
via the blueprint.

Custom steps defined here:
- remove_head: Remove model head up to first LayerNorm
- remove_tail: Remove model tail after second output
- generate_reference_io: Generate reference inputs/outputs for validation

Core brainsmith steps also used in bert_demo.yaml:
- bert_cleanup, bert_streamlining: from brainsmith.steps.bert_steps
- shell_metadata_handover: from brainsmith.steps.bert_steps

These steps are highly specific to BERT model architecture and demonstrate
how to create example-specific steps using the @step decorator without
needing a full package structure.
"""

import logging

import finn.core.onnx_exec as oxe
import numpy as np
from qonnx.core.datatype import DataType
from qonnx.transformation.general import GiveReadableTensorNames, RemoveUnusedTensors
from qonnx.util.basic import gen_finn_dt_tensor

from brainsmith.registry import step

logger = logging.getLogger(__name__)


@step(name="remove_head")
def remove_head_step(model, cfg):
    """Remove all nodes up to the first LayerNormalization node and rewire input."""

    assert len(model.graph.input) == 1, "Error the graph has more inputs than expected"
    {output: node for node in model.graph.node for output in node.output}

    to_remove = []

    current_tensor = model.graph.input[0].name
    current_node = model.find_consumer(current_tensor)
    while current_node.op_type != "LayerNormalization":
        to_remove.append(current_node)
        assert len(current_node.output) == 1, "Error expected an linear path to the first LN"
        current_tensor = current_node.output[0]
        current_node = model.find_consumer(current_tensor)

    # Send the global input to the consumers of the layernorm output
    LN_output = current_node.output[0]
    consumers = model.find_consumers(LN_output)

    # Remove nodes
    to_remove.append(current_node)
    for node in to_remove:
        model.graph.node.remove(node)

    in_vi = model.get_tensor_valueinfo(LN_output)
    model.graph.input.pop()
    model.graph.input.append(in_vi)
    model.graph.value_info.remove(in_vi)

    # Reconnect input
    for con in consumers:
        for i, ip in enumerate(con.input):
            if ip == LN_output:
                con.input[i] = model.graph.input[0].name

    # Clean up after head removal
    for transform in [RemoveUnusedTensors(), GiveReadableTensorNames()]:
        model = model.transform(transform)

    return model


def _recurse_model_tail_removal(model, to_remove, node):
    """Helper function for recursively walking the BERT graph from the second
    output up to the last LayerNorm to remove it"""
    if node is not None:
        if node.op_type != "LayerNormalization":
            to_remove.append(node)
            for tensor in node.input:
                _recurse_model_tail_removal(model, to_remove, model.find_producer(tensor))
    return


@step(name="remove_tail")
def remove_tail_step(model, cfg):
    """Remove from global_out_1 all the way back to the first LayerNorm."""
    # Direct implementation from old custom_step_remove_tail
    out_names = [x.name for x in model.graph.output]
    assert (
        "global_out_1" in out_names
    ), "Error: expected one of the outputs to be called global_out_1, we might need better pattern matching logic here"

    to_remove = []
    current_node = model.find_producer("global_out_1")
    _recurse_model_tail_removal(model, to_remove, current_node)

    for node in to_remove:
        model.graph.node.remove(node)
    del model.graph.output[out_names.index("global_out_1")]

    return model


@step(name="generate_reference_io")
def generate_reference_io_step(model, cfg):
    """
    This step is to generate a reference IO pair for the
    onnx model where the head and the tail have been
    chopped off.
    """
    input_m = model.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)
    np.save(cfg.output_dir + "/input.npy", in_tensor)

    input_t = {input_m.name: in_tensor}
    out_name = model.graph.output[0].name

    y_ref = oxe.execute_onnx(model, input_t, True)
    np.save(cfg.output_dir + "/expected_output.npy", y_ref[out_name])
    np.savez(cfg.output_dir + "/expected_context.npz", **y_ref)
    return model
