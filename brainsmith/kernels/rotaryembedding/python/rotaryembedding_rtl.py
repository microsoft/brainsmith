# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
import shutil

import numpy as np
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)
from finnbrainsmith.custom_op.fpgadataflow.rotaryembedding import RotaryEmbedding
from qonnx.util.basic import roundup_to_integer_multiple

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


class RotaryEmbedding_rtl(RotaryEmbedding, RTLBackend):
    """CustomOp wrapper for the finn-rtllib RotaryEmbedding_axi component
    Supports adjusting the padding amount and spatial feature sizes at
    runtime."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(RotaryEmbedding.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_verilog_top_module_intf_names(self):
        # Overload default HLSCustomOp implementation to add axilite control IF
        intf_names = super().get_verilog_top_module_intf_names()
        # Override the AXIS I/O
        intf_names["s_axis"] = [("in_" + self.hls_sname(), self.get_instream_width_padded())]
        intf_names["m_axis"] = [("out_" + self.hls_sname(), self.get_outstream_width_padded())]
        return intf_names

    def assert_context_has_input(self, context, inp_name):
        assert inp_name in context, "Missing expected input %s" % inp_name

    def assert_datatype_is_float(self, dt):
        assert str(dt) == "float32", "Only float32 datatype supported"

    def assert_expected_shape(self, shape, exp_shape, help_explain_shape=""):
        assert shape == exp_shape, (
            f"Expected Shape {exp_shape} Actual Shape: {shape}" + f"\n{help_explain_shape}"
        )

    def validate_input(self, context, name):
        self.assert_context_has_input(context, name)
        self.assert_datatype_is_float(context[name].dtype)
        self.assert_expected_shape(
            context[name].shape,
            self.get_normal_input_shape(),
            "Input Shape (1, NumberOfHeads, SequenceLength, HiddenDimension)",
        )

    def validate_output(self, context, otensor):
        self.assert_context_has_input(context, otensor)
        self.assert_expected_shape(
            context[otensor],
            self.get_normal_output_shape(),
            "Output Shape (1, NumberOfHeads, SequenceLength, HiddenDimension)",
        )

    def export_to_npy(self, directory, filename, itensor):
        np.save(os.path.join(directory, filename), itensor)

    def import_io_from_onnx_node(self):
        io_dict = {}
        for inp in self.onnx_node.input:
            io_dict[inp] = []
        for outp in self.onnx_node.output:
            io_dict[outp] = []
        return io_dict

    def import_test_vectors_from_context(self, context, name_to_data_dict):
        for name in name_to_data_dict:
            self.validate_input(context, name_to_data_dict[name])
            name_to_data_dict[name].append(context[name])

    def export_outputs_to_context(self, context, name_to_data_dict):
        for name in name_to_data_dict:
            self.validate_output(context, name)
            context[name] = name_to_data_dict[name]

    def apply_folding(self, name_to_data_dict):
        for name in name_to_data_dict:
            name_to_data_dict[name] = name_to_data_dict[name].reshape(self.get_folded_input_shape())
        return name_to_data_dict

    def apply_output_unfolding(self, name_to_data_dict):
        for name in name_to_data_dict:
            name_to_data_dict[name] = name_to_data_dict[name].reshape(
                self.get_normal_output_shape()
            )
        return name_to_data_dict

    def convert_npy_to_rtlsim(self, name_to_data_dict):
        for name in name_to_data_dict:
            name_to_data_dict[name] = npy_to_rtlsim_input(
                name_to_data_dict[name], self.get_input_datatype(), self.get_instream_width()
            )
        return name_to_data_dict

    def convert_rtlsim_to_npy(self, name_to_data_dict):
        for name in name_to_data_dict:
            name_to_data_dict[name] = rtlsim_output_to_npy(
                name_to_data_dict[name],
                None,
                self.get_output_datatype(),
                self.get_folded_output_shape(),
                self.get_outstream_width(),
                self.get_outstream_width(),
                self.get_output_datatype().bitwidth(),
            )
        return name_to_data_dict

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim":
            # Prepare Input For RTL Simulation
            io_dict = self.import_io_from_onnx_node()
            io_dict["inputs"] = self.import_test_vectors_from_context(context, io_dict["inputs"])
            io_dict["inputs"] = self.apply_folding(io_dict["inputs"])
            io_dict["inputs"] = self.convert_npy_to_rtlsim(io_dict["inputs"])

            # Run RTL Simulation
            sim = self.get_rtlsim()
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            self.rtlsim_multi_io(sim, io_dict)

            # Process RTL Simulation Output
            io_dict["outputs"] = self.convert_rtlsim_to_npy(io_dict["outputs"])
            io_dict["outputs"] = self.apply_output_unfolding(io_dict["outputs"])
            self.export_outputs_to_context(context, io_dict["outputs"])

        else:
            raise Exception(
                f"""Invalid value for attribute exec_mode! Is currently set to: {mode}
            has to be set to one of the following value ("cppsim", "rtlsim")"""
            )

    def get_template_values(self, head_dim, seq_len, hidden, simd, idt, wdt):
        topname = self.get_verilog_top_module_name()
        stream_bits = idt.bitwidth() * simd
        stream_bits = int(roundup_to_integer_multiple(stream_bits, 8))
        code_gen_dict = {
            "HEAD_DIM": int(head_dim),
            "SEQ_LEN": int(seq_len),
            "HIDDEN_DIM": int(hidden),
            "SIMD": int(simd),
            "ELEM_BITS": idt.bitwidth(),
            "WEIGHT_BITS": wdt.bitwidth(),
            "TOP_MODULE_NAME": topname,
            "STREAM_BITS": int(stream_bits),
            "COS_INIT_FILE": '"cos_values.dat"',
            "SIN_INIT_FILE": '"sin_values.dat"',
        }
        return code_gen_dict

    def get_dynamic_config(self, ifm_dims=None, pads=None):
        raise NotImplementedError("This Method is not implemented")
        # """Returns a configuration dict to re-configure FM dimension and
        # padding amounts during runtime."""

        # if ifm_dims is None:
        #     ifm_dims = self.get_nodeattr("ImgDim")
        # if pads is None:
        #     pads = self.get_nodeattr("Padding")
        # chans = self.get_nodeattr("NumChannels")
        # simd = self.get_nodeattr("SIMD")
        # idt = self.get_input_datatype()
        # code_gen_dict = self.get_template_values(ifm_dims, pads, chans, simd, idt)
        # config = {
        #     "XON": (0 * 4, (code_gen_dict["INIT_XON"])),
        #     "XOFF": (1 * 4, (code_gen_dict["INIT_XOFF"])),
        #     "XEND": (2 * 4, (code_gen_dict["INIT_XEND"])),
        #     "YON": (3 * 4, (code_gen_dict["INIT_YON"])),
        #     "YOFF": (4 * 4, (code_gen_dict["INIT_YOFF"])),
        #     "YEND": (5 * 4, (code_gen_dict["INIT_YEND"])),
        # }
        # return config

    def generate_hdl(self, model, fpgapart, clk):
        head_dim = self.get_nodeattr("HeadDimension")
        hidden = self.get_nodeattr("HiddenDimension")
        seq_len = self.get_nodeattr("SequenceLength")
        simd = self.get_nodeattr("SIMD")
        idt = self.get_input_datatype()
        wdt = self.get_weight_datatype()
        code_gen_dict = self.get_template_values(head_dim, seq_len, hidden, simd, idt, wdt)
        self.make_weight_file(
            "cos_values.dat",
            wdt,
            model.get_initializer("cos_quant"),
        )
        self.make_weight_file(
            "sin_values.dat",
            wdt,
            model.get_initializer("sin_quant"),
        )
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        finnbrainsmith_rtllib = os.environ["FINN_ROOT"] + "/deps/finnbrainsmith/rtllib/rope/hdl"
        finn_rtllib = os.environ["FINN_ROOT"] + "/finn-rtllib"

        sv_files = []
        sv_files.append(finnbrainsmith_rtllib + "/rope_axi.sv")
        sv_files.append(finnbrainsmith_rtllib + "/rope.sv")
        sv_files.append(finn_rtllib + "/memstream/hdl/memstream.sv")
        sv_files.append(finn_rtllib + "/fifo/hdl/Q_srl.v")

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(finnbrainsmith_rtllib + "/rope_template.v") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(
            os.path.join(code_gen_dir, self.get_verilog_top_module_name() + ".v"),
            "w",
        ) as f:
            f.write(template)

        for sv_file in sv_files:
            shutil.copy(sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""
        # Modified to use generated (System-)Verilog instead of HLS output products

        # if PyVerilator is None:
        #      raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]
        verilog_files = [
            "rope_axi.sv",
            "rope.sv",
            "memstream.sv",
            "Q_srl.v",
            self.get_nodeattr("gen_top_module") + ".v",
        ]

        # ret = pyxsi_utils.compile_sim_obj(
        #        self.get_verilog_top_module_name(), verilog_files, code_gen_dir
        # )
        # save generated lib filename in attribute
        # self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

        # build the Verilator emu library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_verilog_top_module_name(),
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return None

    def code_generation_ipi(self, behavioral=False):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "rope_axi.sv",
            "rope.sv",
            "memstream.sv",
            "Q_srl.v",
            self.get_nodeattr("gen_top_module") + ".v",
        ]

        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference {} {}".format(
                self.get_nodeattr("gen_top_module"), self.onnx_node.name
            )
        ]
        return cmd

    def make_weight_file(self, weight_file_name, wdt, weights):
        weight_stream = []

        simd = self.get_nodeattr("SIMD")
        bw_hexdigit = simd * wdt.bitwidth()
        print("bw_hexdigit: ", bw_hexdigit)
        print("weights ", weights)
        # iterate over the file weight dimension

        for seq in weights[0, 0, :]:
            if simd > 1:
                seq = seq.reshape(-1, simd)
            for w in seq:
                t_packed = pack_innermost_dim_as_hex_string(
                    [np.flip(w)], wdt, bw_hexdigit, prefix=""
                ).item()
                weight_stream.append(t_packed)

            # t_packed = pack_innermost_dim_as_hex_string(
            #                    [w], wdt, bw_hexdigit, prefix=""
            #                ).item()
            # weight_stream.append(t_packed)
        print("weight file name:", weight_file_name)
        with open(weight_file_name, "w") as f:
            for val in weight_stream:
                f.write(val + "\n")

    def get_rtl_file_list(self):
        pass
