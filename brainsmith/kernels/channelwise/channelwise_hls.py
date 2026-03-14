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
#
# Modified by Microsoft Corporation for integration with Brainsmith toolchain.
# Modifications licensed under the MIT License.

from math import ceil

import numpy as np
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import numpy_to_hls_code
from qonnx.core.datatype import DataType

from brainsmith.kernels.channelwise.channelwise import ChannelwiseOp
from brainsmith.registry import backend


@backend(
    name="ChannelwiseOpHLS",
    target_kernel="brainsmith:ChannelwiseOp",
    language="hls",
    author="FINN Team",
)
class ChannelwiseOp_hls(ChannelwiseOp, HLSBackend):
    """HLS backend for ChannelwiseOp (KernelOp-based)."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Combine ChannelwiseOp and HLSBackend nodeattrs."""
        my_attrs = ChannelwiseOp.get_nodeattr_types(self)
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    # Removed _get_mem_mode() and get_instream_width() - embedded mode only, use base class

    # ================================================================
    # Resource Estimation (Uses design_point)
    # ================================================================

    def bram_estimation(self):
        """Calculates BRAM cost if resource set to BRAM."""
        style = self.get_nodeattr("ram_style")
        input_iface = self.design_point.inputs["input"]
        pe = input_iface.stream_shape[-1]  # PE from stream tiling
        A = self.get_input_datatype(0).bitwidth()
        tmem = self.calc_tmem()  # Uses design_point

        if style == "block" and tmem > 1:
            return int(ceil(A * pe / 16)) * int(ceil(tmem / 1024))
        else:
            return 0

    def lut_estimation(self):
        """Calculates LUT cost, taking memory resource type into account."""
        style = self.get_nodeattr("ram_style")
        input_iface = self.design_point.inputs["input"]
        pe = input_iface.stream_shape[-1]  # PE from stream tiling
        A = self.get_input_datatype(0).bitwidth()
        tmem = self.calc_tmem()

        # Cost of comparators/operators
        comparator_cost = A * pe

        # Cost of LUTRAM
        if style == "distributed" and tmem > 1:
            lutram_cost = pe * A * int(ceil(tmem / 64))
        else:
            lutram_cost = 0

        return comparator_cost + lutram_cost

    # ================================================================
    # HLS Code Generation
    # ================================================================

    def get_template_param_values(self):
        """Returns template parameter values for HLS code."""
        ret = dict()
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        ret["TSrcI"] = f"Slice<{inp_hls_str}>"
        ret["TDstI"] = f"Slice<{out_hls_str}>"
        return ret

    def generate_params(self, model, path):
        """Generate parameter header file (embedded mode only)."""
        code_gen_dir = path
        parameters = model.get_initializer(self.onnx_node.input[1])
        # Embedded mode: generate params.h header file
        weight_filename = f"{code_gen_dir}/params.h"
        self.make_weight_file(parameters, "hls_header", weight_filename)

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce file containing parameters in HLS header format.

        Args:
            weights: numpy array with parameters
            weight_file_mode: must be "hls_header" (embedded mode only)
            weight_file_name: filename for weight file
        """
        if weight_file_mode != "hls_header":
            raise Exception(f"Only hls_header mode supported (got {weight_file_mode})")

        parameter_tensor = self.get_hls_compatible_parameter_tensor(weights)
        pdt = DataType[self.get_input_datatype(1).name]

        # Generate params.h with ChannelWiseOperation initializer
        parameters_hls_code = numpy_to_hls_code(
            parameter_tensor, pdt, "parameters", False, True
        )

        # Get datatypes
        idt = self.get_input_datatype(0)
        if idt == DataType["BIPOLAR"]:
            idt = DataType["BINARY"]
        idt_hls = idt.get_hls_datatype_str()

        pdt_hls = pdt.get_hls_datatype_str()

        odt = self.get_output_datatype()
        if odt == DataType["BIPOLAR"]:
            odt = DataType["BINARY"]
        odt_hls = odt.get_hls_datatype_str()

        # Get operation function and map to HLS templates
        func = self.get_nodeattr("func")
        func_map = {
            "LessOrEqual": f"comp::less_equal<{idt_hls}, {pdt_hls}>",
            "GreaterOrEqual": f"comp::greater_equal<{idt_hls}, {pdt_hls}>",
            "Add": f"comp::add<{odt_hls}, {odt_hls}, {odt_hls}>",
            "Mul": f"comp::mul<{odt_hls}, {odt_hls}, {odt_hls}>",
        }

        if func not in func_map:
            raise Exception(
                f"Invalid value for attribute func! Is currently set to: {func}. "
                f"Must be one of: {list(func_map.keys())}"
            )

        func_str = func_map[func]
        input_iface = self.design_point.inputs["input"]
        pe = input_iface.stream_shape[-1]  # PE from stream tiling
        tmem = self.calc_tmem()

        # Write params.h
        with open(weight_file_name, "w") as f:
            f.write(
                f"static ChannelWiseOperation<{tmem},{pe},{idt_hls},"
                f"{pdt_hls},{odt_hls},{func_str}> threshs = "
            )
            f.write(parameters_hls_code)

    # NOTE: get_rtlsim() and execute_node() inherited from HWCustomOp/HLSBackend
    # No overrides needed - FINN's implementation works correctly!

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"', '#include "params.h"']

    def defines(self, var):
        # Use design_point for semantic shape (not nodeattrs)
        input_iface = self.design_point.inputs["input"]
        tensor_shape = input_iface.tensor_shape
        num_channels = tensor_shape[-1]  # Last dimension is channels (NHWC)
        numReps = tensor_shape[0]  # First dimension is batch
        pe = input_iface.stream_shape[-1]  # PE from stream tiling

        # Use multi-line string with actual newlines (not escaped \n)
        self.code_gen_dict["$DEFINES$"] = [
            f"""#define NumChannels1 {num_channels}
#define PE1 {pe}
#define numReps {numReps}"""
        ]

    def read_npy_data(self):
        """Read NPY data for input stream (embedded mode only)."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype(0)
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = f"{code_gen_dir}/input_0.npy"

        self.code_gen_dict["$READNPYDATA$"] = [
            f"npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, "
            f'{npy_type}>("{npy_in}", in0_V, false);'
        ]

    def strm_decl(self):
        """Generate stream declarations (embedded mode only)."""
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            f'hls::stream<ap_uint<{self.get_instream_width(0)}>> in0_V ("in0_V");',
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");',
        ]

    def docompute(self):
        """Generate compute code (embedded mode only, matches FINN implementation)."""
        tmpl_args = self.get_template_param_values()

        # Spatial dim from semantic NHWC shape (design_point)
        block_shape = self.design_point.inputs["input"].block_shape

        # NHWC format guaranteed by schema
        if len(block_shape) == 4:  # [N, H, W, C]
            spatial_dim = block_shape[1] * block_shape[2]
        elif len(block_shape) == 2:  # [N, C] - fully connected
            spatial_dim = 1
        else:
            raise Exception(f"Unexpected block_shape {block_shape}")

        # Embedded mode: parameters from params.h (FINN parity)
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"Thresholding_Batch<{spatial_dim}, NumChannels1, PE1, "
            f"{tmpl_args['TSrcI']}, {tmpl_args['TDstI']}>"
            f"(in0_V, out0_V, threshs, numReps);"
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            dtype = DataType["BINARY"]

        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = f"{code_gen_dir}/output_0.npy"
        # Folded shape for HLS I/O (FINN-specific format)
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f"apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, "
            f'{npy_type}>(out0_V, {shape_cpp_str}, "{npy_out}", false);'
        ]

    def blackboxfunction(self):
        """Generate blackbox function signature (embedded mode only)."""
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}(hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V, "
            f"hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V)"
        ]

    def pragmas(self):
        """Generate HLS pragmas (embedded mode only)."""
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            "#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=1",
        ]

        # Set resource type for parameter storage
        ram_style = self.get_nodeattr("ram_style")
        input_iface = self.design_point.inputs["input"]
        pe = input_iface.stream_shape[-1]  # PE from stream tiling
        ich = input_iface.tensor_shape[-1]  # Total channels

        if pe < ich:
            if ram_style == "distributed":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_LUTRAM"
                )
            elif ram_style == "block":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_BRAM"
                )
            else:
                raise Exception(
                    f"Invalid value for attribute ram_style! Is currently set to: {ram_style}. "
                    f"Must be one of: ('block', 'distributed')"
                )
