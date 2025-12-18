############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
############################################################################

from typing import Tuple
import pytest
import torch
import onnx
import torch.nn as nn
import brevitas.nn as qnn
import finn.core.onnx_exec as oxe
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.transformation.infer_datatypes import InferDataTypes
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias

# Debugging dependencies, to remove
import os

from qonnx.transformation.fold_constants import FoldConstants

from qonnx.transformation.general import GiveUniqueNodeNames
from finn.transformation.general import ApplyConfig

import numpy as np

test_fpga_part = "xcv80-lsva4737-2MHP-e-s"
target_clk_ns = 5

def create_layernorm_model(epsilon):

    tshape = [1, 128, 384]
    scale_bias_shape = tshape[-1]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, tshape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, tshape)
    LayerNorm_scale = helper.make_tensor_value_info("LayerNorm_Scale", TensorProto.FLOAT, [scale_bias_shape])
    LayerNorm_bias = helper.make_tensor_value_info("LayerNorm_Bias", TensorProto.FLOAT, [scale_bias_shape])

    ln_node = helper.make_node(
        'LayerNormalization',
        inputs=["inp", "LayerNorm_Scale", "LayerNorm_Bias"],
        outputs=["outp"],
        name='Layernorm_0',
        epsilon=epsilon,
        axis=-1,
        stash_type=1,
    )

    # Create model
    graph = helper.make_graph(
        nodes=[ln_node], name="LayerNorm_graph", inputs=[inp], outputs=[outp]
    )
    model = qonnx_make_model(graph, producer_name="LayerNorm_graph")
    model = ModelWrapper(model)

    # Tensor initializers
    max_scale = 2**(8/2)
    max_bias = 2**(8/2)
    model.set_initializer("LayerNorm_Scale", (max_scale*np.random.rand(scale_bias_shape)).astype(np.float32))
    model.set_initializer("LayerNorm_Bias", (max_bias*np.random.rand(scale_bias_shape)).astype(np.float32))

    return model

def test_fpgadataflow_layernorm():
    model = create_layernorm_model(epsilon=9.999999960041972e-13)

    # reference calculation
    input = gen_finn_dt_tensor(DataType["FLOAT32"], [1, 128, 384])
    input_t = {model.graph.input[0].name: input}

    y_ref = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    model = model.transform(ExtractNormScaleBias())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(to_hw.InferLayerNorm())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    getCustomOp(model.graph.node[0]).set_nodeattr("SIMD", 8)

    # Execute
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    input_t = {model.graph.input[0].name: input}

    y_hw = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    assert np.allclose(y_ref, y_hw, rtol=1e-3, atol=2**-4)

    # stitched ip rtlsim
    model = model.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))

    model.set_metadata_prop("exec_mode", "rtlsim")
    input_t = {model.graph.input[0].name: input}
    y_stitch = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    assert np.allclose(y_ref, y_stitch, rtol=1e-3, atol=2**-4)
