# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
import numpy as np
import onnx
import onnx.helper as helper
import pytest
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
from finn.util.basic import pynq_part_map
from finnbrainsmith.custom_op.fpgadataflow.rotaryembedding import get_rope_onnx_filename
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from finn.transformation.general import ApplyConfig
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


class QOnnxQuantizeNodeConfig:
    def __init__(self, scale=1.0, zeropt=0, bitwidth=8, narrow=0, signed=1, rounding_mode="ROUND"):
        self.narrow = narrow
        self.signed = signed
        self.rounding_mode = rounding_mode
        self.scale = scale
        self.zeropt = zeropt
        self.bitwidth = bitwidth


def make_quant_node(name, inp, outp, qconfig):
    return helper.make_node(
        "Quant",
        domain="qonnx.custom_op.general",
        inputs=[inp.name, f"{name}_quant_scale", f"{name}_quant_zeropt", f"{name}_quant_bitwidth"],
        outputs=[outp.name],
        narrow=qconfig.narrow,
        signed=qconfig.signed,
        rounding_mode=qconfig.rounding_mode,
        name=f"{name}QuantNode",
    )


def get_tensorinfo_shape(itensor):
    return [dim.dim_value for dim in itensor.type.tensor_type.shape.dim]


def add_quant_node_to_tensor(model, name, itensor, qconfig, as_consumer=True):
    # Create a tensor output for the new quantize node
    ntensor = helper.make_tensor_value_info(
        f"{name}_quant_0", TensorProto.FLOAT, get_tensorinfo_shape(itensor)
    )

    # Create the new quantize node
    if as_consumer:
        node = make_quant_node(name, itensor, ntensor, qconfig)
    else:
        node = make_quant_node(name, ntensor, itensor, qconfig)

    # update the graph
    model.graph.node.append(node)
    model.graph.value_info.append(ntensor)

    # update the initializers
    model.set_initializer(node.input[1], np.asarray([qconfig.scale], dtype=np.float32))
    model.set_initializer(node.input[2], np.asarray([qconfig.zeropt], dtype=np.float32))
    model.set_initializer(node.input[3], np.asarray([qconfig.bitwidth], dtype=np.float32))

    return ntensor


def add_quant_node_before_tensor(model, name, otensor, qconfig):
    # Create a tensor input for the new quantize node
    itensor = helper.make_tensor_value_info(
        f"{name}_quant_input_0", TensorProto.FLOAT, get_tensorinfo_shape(otensor)
    )

    # Create the new quantize node
    node = make_quant_node(name, itensor, otensor, qconfig)

    # update the graph
    model.graph.node.append(node)
    model.graph.value_info.append(itensor)

    # update the initializers
    model.set_initializer(node.input[1], np.asarray([qconfig.scale], dtype=np.float32))
    model.set_initializer(node.input[2], np.asarray([qconfig.zeropt], dtype=np.float32))
    model.set_initializer(node.input[3], np.asarray([qconfig.bitwidth], dtype=np.float32))

    return itensor


def display_errors_from_binary_operation(
    expected, c, a, b, a_mask=[0, 0, 0, 0], b_mask=[0, 0, 0, 0], c_mask=[0, 0, 0, 0], desc=None
):
    mismatched = np.asarray(expected != c).nonzero()

    a_indices = list(mismatched)
    b_indices = list(mismatched)
    c_indices = list(mismatched)

    for i in range(len(mismatched)):
        a_indices[i] = 0 if a_mask[i] == 1 else mismatched[i]
        b_indices[i] = 0 if b_mask[i] == 1 else mismatched[i]
        c_indices[i] = 0 if c_mask[i] == 1 else mismatched[i]

    np.set_printoptions(suppress=True)
    if desc is not None:
        print(f"{desc}")
    print(f"mismatched: {mismatched}")
    print(f"a: {a[tuple(a_indices)]}")
    print(f"b: {b[tuple(b_indices)]}")
    print(f"c: {c[tuple(c_indices)]}")
    print(f"expected: {expected[tuple(c_indices)]}")


def make_single_rope_modelwrapper(
    seq_len, hidden, head_size, num_heads, idt, wdt, cos_quant, sin_quant, simd, impl_style
):
    io_shape = [1, num_heads, seq_len, head_size]

    # Define the input tensor
    act_in = helper.make_tensor_value_info("act_in", onnx.TensorProto.FLOAT, io_shape)

    # Define the output tensor
    act_out = helper.make_tensor_value_info("act_out", onnx.TensorProto.FLOAT, io_shape)

    # Create the graph
    graph = helper.make_graph(
        nodes=[],
        name="RopeGraph",
        inputs=[act_in],
        outputs=[act_out],
        initializer=[
            helper.make_tensor("cos_quant", onnx.TensorProto.FLOAT, cos_quant.shape, cos_quant),
            helper.make_tensor("sin_quant", onnx.TensorProto.FLOAT, sin_quant.shape, sin_quant),
        ],
    )

    # Create the QONNX model
    model = qonnx_make_model(graph, producer_name="rope-model")
    model = ModelWrapper(model)

    IQuantConfig = QOnnxQuantizeNodeConfig(bitwidth=idt.bitwidth())
    act_quant = add_quant_node_to_tensor(model, "act_in", act_in, IQuantConfig)

    # Calculate the scale and zero point for the SIN/COS quantize nodes
    # sincos_beta  = 1.0
    # sincos_alpha = -sincos_beta
    # sincos_qbeta =  2 ** (wdt.bitwidth()-2)
    # sincos_qalpha = -sincos_qbeta
    # sincos_scale = (sincos_beta - sincos_alpha) / (sincos_qbeta - sincos_qalpha)
    # sincos_zeropt = -((sincos_alpha/sincos_scale) - sincos_qalpha)

    # print(f"beta-alpha:{sincos_beta-sincos_alpha} qbeta-qalpha: {sincos_qbeta - sincos_qalpha} scale: {sincos_scale}, zeropt: {sincos_zeropt}")

    # WQuantConfig = QOnnxQuantizeNodeConfig(scale=sincos_scale, zeropt=sincos_zeropt, narrow=1.0, bitwidth=wdt.bitwidth())
    # cosTVI = helper.make_tensor_value_info('cos', onnx.TensorProto.FLOAT, model.get_initializer("cos").shape)
    # sinTVI = helper.make_tensor_value_info('sin', onnx.TensorProto.FLOAT, model.get_initializer("sin").shape)

    # cos_quant = add_quant_node_to_tensor(model, "cos", cosTVI, WQuantConfig)
    # sin_quant = add_quant_node_to_tensor(model, "sin", sinTVI, WQuantConfig)

    cos_quant_tv = helper.make_tensor_value_info(
        "cos_quant", onnx.TensorProto.FLOAT, cos_quant.shape
    )
    sin_quant_tv = helper.make_tensor_value_info(
        "sin_quant", onnx.TensorProto.FLOAT, sin_quant.shape
    )

    # Define the custom RoPE node
    rope_node = helper.make_node(
        "RotaryEmbedding",  # Custom node name
        [act_quant.name, cos_quant_tv.name, sin_quant_tv.name],  # Inputs
        #  cos_quant_otensor.name, sin_quant_otensor.name],  # Inputs],
        ["act_out"],  # Outputs
        name="CustomRoPE",
        domain="finnbrainsmith.custom_op.fpgadataflow",
        backend="fpgadataflow",
        HiddenDimension=hidden,
        SequenceLength=seq_len,
        HeadDimension=head_size,
        NumHeads=num_heads,
        RopeTheta=10000.0,
        inputDataType=str(idt.name),
        weightDataType=str(wdt.name),
        numInputVectors=1,
        SIMD=simd,
        preferred_impl_style=impl_style,
    )

    # Add the custom node to the graph
    model.graph.node.append(rope_node)

    model.set_metadata_prop("rtlsim_trace", "trace.vcd")
    os.environ["RTLSIM_TRACE_DEPTH"] = "45"

    # Save the model to a file
    model.save("rope_node.onnx")

    return model


# input image dimension
# @pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of channels
@pytest.mark.parametrize("seq_len", [2, 4, 32, 128])
@pytest.mark.parametrize("head_size", [32, 64])
@pytest.mark.parametrize("num_heads", [8])
# Input parallelism
@pytest.mark.parametrize("simd", [1, 2, 4, 8, 16, 32])
# FINN input datatype
@pytest.mark.parametrize("idt", [DataType["INT8"], DataType["INT16"]])
@pytest.mark.parametrize("wdt", [DataType["INT16"]])
# execution mode
# @pytest.mark.parametrize("mode", ["cppsim", "rtlsim"])
# implementation style
@pytest.mark.parametrize("impl_style", ["rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_rope(seq_len, head_size, num_heads, idt, wdt, simd, impl_style):
    hidden = head_size * num_heads

    if simd >= head_size:
        pytest.skip(
            "The current implementation does not support an SIMD == or greater than head_size. This can likely be fixed but haven't had the time to do so yet."
        )

    assert head_size % simd == 0

    def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

    assert is_power_of_two(simd)

    act_in = gen_finn_dt_tensor(idt, [1, num_heads, seq_len, head_size])
    print(f"act_in: {act_in}")
    # genn_finn_dt_tensor generates positive 128 values for INT8 which are not within its range
    act_in = np.where(act_in > idt.max(), idt.max(), act_in)
    act_in = np.where(act_in <= idt.min(), idt.min() + 1, act_in)

    midpoint = head_size // 2

    act_in1 = np.concatenate((-act_in[..., midpoint:], act_in[..., :midpoint]), axis=-1)

    # open onnx file and retrieve weights
    onnx_filename = get_rope_onnx_filename(10000.0, 1, num_heads, seq_len, head_size)
    onnx_path = (
        "../../src/finnbrainsmith/custom_op/fpgadataflow/rotaryembedding/onnxgraphs/"
        + onnx_filename
    )
    print(f"pwd: {os.getcwd()}")
    onnx_model = onnx.load(onnx_path)

    qonnx_model = ModelWrapper(onnx_model)
    cos = np.expand_dims(qonnx_model.get_initializer("cos_param"), 0)
    sin = np.expand_dims(qonnx_model.get_initializer("sin_param"), 0)

    sincos_quant_scale = 2 ** (wdt.bitwidth() - 2)
    cos_quant = np.round(cos * sincos_quant_scale)
    sin_quant = np.round(sin * sincos_quant_scale)

    # The multiplication of the q, cos_quant and q1, sin_quant values are lossless in the hardware
    # Promote the values to float64 to avoid rounding errors with FP32
    cos_mul = np.round(np.float64(act_in) * np.float64(cos_quant) / sincos_quant_scale)
    sin_mul = np.round(np.float64(act_in1) * np.float64(sin_quant) / sincos_quant_scale)

    expected = cos_mul + sin_mul

    clip = 2 ** (idt.bitwidth() - 1) - 1
    expected = np.where(expected > clip, clip, expected)
    expected = np.where(expected < -clip, -clip, expected)
    expected = np.where(expected == -0.0, 0.0, expected)
    input_dict = {"input": act_in}

    model = make_single_rope_modelwrapper(
        seq_len, hidden, head_size, num_heads, idt, wdt, cos_quant, sin_quant, simd, impl_style
    )

    # model.save("rope_model-before-convert-qonnx-to-finn.onnx")
    model = model.transform(
        ConvertQONNXtoFINN(
            filter_function=default_filter_function_generator(
                max_multithreshold_bit_width=idt.bitwidth()
            )
        )
    )
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model.save("rope_model-after-convert-qonnx-to-finn.onnx")

    # model.save("rope_model-before-infer-shapes.onnx")
    model = model.transform(InferShapes())

    # model = model.transform(RoundAndClipThresholds())
    # model.save("rope_model-before-infer-thresholding-layer.onnx")
    model = model.transform(to_hw.InferThresholdingLayer())
    model.save("rope_model-after-infer-thresholding-layer.onnx")

    # Isolate fpga dataflow layers
    # parent_model = model.transform(CreateDataflowPartition())
    # parent_model.save('parent_model.onnx') # Debug
    # sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    # sdp_node_path = getCustomOp(sdp_node).get_nodeattr("model")
    # model = ModelWrapper(sdp_node_path)
    # model.save('partitioned_model.onnx') # Debug

    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model.save("rope_model-after-specialize.onnx")

    # write a json SIMD configuration file
    config = {
        "Defaults": {},
        "Thresholding_rtl_0": {
            "PE": simd,
            "runtime_writeable_weights": 0,
            "depth_trigger_uram": 0,
            "depth_trigger_bram": 0,
        },
    }
    import json

    with open("simd_config.json", "w") as f:
        json.dump(config, f)

    model = model.transform(ApplyConfig("simd_config.json"))

    model = model.transform(SetExecMode("rtlsim"))
    # model.set_metadata_prop('rtlsim_backend', "pyxsi")
    # model.set_metadata_prop("exec_mode", "rtlsim")
    print("prepare ip")
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    print("hls synth ip")
    model = model.transform(HLSSynthIP())
    print("prepare rtlsim")
    model = model.transform(PrepareRTLSim())
    print("create stitched ip")
    model.save("rope_model-before-create-stitched-ip.onnx")
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))

    model.save("rope_model-after-create-stitched-ip.onnx")

    model.set_metadata_prop("exec_mode", "rtlsim")
    model.set_metadata_prop("rtlsim_backend", "pyverilator")
    input_dict = {"act_in": act_in}
    sim_output = oxe.execute_onnx(model, input_dict)

    display_errors_from_binary_operation(
        expected, sim_output["act_out"], act_in, cos_quant, b_mask=[1, 1, 0, 0], desc="cos-mul"
    )
    display_errors_from_binary_operation(
        expected, sim_output["act_out"], act_in1, sin_quant, b_mask=[1, 1, 0, 0], desc="sin-mul"
    )
    display_errors_from_binary_operation(
        expected, sim_output["act_out"], cos_mul, sin_mul, desc="add"
    )

    assert (sim_output["act_out"] == expected).all()

    op_type = "RotaryEmbedding_" + "rtl"
    model.save("rope_model-after-specialization.onnx")
    node = model.get_nodes_by_op_type(op_type)[0]
    inst = getCustomOp(node)
    inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles_dict[node.name]
