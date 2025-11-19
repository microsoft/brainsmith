############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
############################################################################

import os
import textwrap
from math import ceil, log2

import numpy as np
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from brainsmith.kernels.thresholding.thresholding import Thresholding
from brainsmith.registry import backend


@backend(
    target_kernel="brainsmith:Thresholding",
    language="hls",
    description="HLS implementation of Thresholding",
    author="Microsoft Corporation",
)
class Thresholding_hls(Thresholding, HLSBackend):
    """HLS backend for Thresholding kernel (KernelOp-based).

    This backend adapts the schema-driven Thresholding implementation
    to work with FINN's HLS code generation system.

    Key features:
    - Extracts shapes from design_point (not nodeattrs)
    - Supports three memory modes (via input1MemType DSE parameter):
      * embedded: Thresholds in thresh.h header (compile-time constant)
      * decoupled: Thresholds in separate memory, streamed via in1_V
      * dynamic: Thresholds streamed from external source (MLO mode)

    Memory Modes:
    - embedded: Thresholds embedded in HLS code (smallest, fastest)
    - decoupled: Thresholds in separate BRAM/LUT, streamed via in1_V
    - dynamic: External streaming (MLO), no internal storage
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def _get_mem_mode(self) -> str:
        """Get memory mode from design point interface.

        Returns:
            Memory mode string: "embedded", "decoupled", or "dynamic"
        """
        thresholds_iface = self.design_point.inputs.get("thresholds")
        return (thresholds_iface.mem_mode if thresholds_iface and thresholds_iface.mem_mode
               else "embedded")

    def get_nodeattr_types(self):
        """Define nodeattrs for Thresholding_hls backend.

        Combines:
        - Thresholding's schema-derived nodeattrs
        - HLSBackend's execution nodeattrs
        - HLS-specific parameters (mem_mode, ram_style)
        """
        my_attrs = Thresholding.get_nodeattr_types(self)
        my_attrs.update(HLSBackend.get_nodeattr_types(self))

        # Add HLS-specific nodeattrs
        my_attrs.update(
            {
                # Memory type for embedded mode (BRAM vs LUT)
                # NOTE: mem_mode now comes from design point interface (input1MemType DSE param)
                "ram_style": ("s", False, "distributed", {"distributed", "block"}),
            }
        )

        return my_attrs

    def bram_estimation(self):
        """Calculates BRAM cost if resource set to BRAM."""
        style = self.get_nodeattr("ram_style")
        P = self.get_nodeattr("PE")
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()
        tmem = self.calc_tmem()

        if style == "block" and tmem > 1:
            return int(ceil(A * P / 16)) * int(ceil(tmem / 1024))
        else:
            return 0

    def lut_estimation(self):
        """Calculates LUT cost, taking memory resource type into account."""
        style = self.get_nodeattr("ram_style")
        P = self.get_nodeattr("PE")
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()
        tmem = self.calc_tmem()

        # Cost of comparators
        comparator_cost = A * P

        # Cost of LUTRAM
        if style == "distributed" and tmem > 1:
            lutram_cost = P * A * int(ceil(tmem / 64))
        else:
            lutram_cost = 0

        # Total cost
        return comparator_cost + lutram_cost

    def get_ap_int_max_w(self):
        """Get maximum ap_int width needed."""
        ap_int_max_w = HLSBackend.get_ap_int_max_w(self)

        # Decoupled and dynamic modes have streaming threshold interface
        if self._get_mem_mode() in ("decoupled", "dynamic"):
            weightstream = self.get_instream_width(1)
            ap_int_max_w = max([weightstream, ap_int_max_w])

        return ap_int_max_w

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates c++ code and tcl script for ip generation."""
        super().code_generation_ipgen(model, fpgapart, clk)

        # Decoupled and dynamic modes need memstream HDL
        if self._get_mem_mode() in ("decoupled", "dynamic"):
            self.generate_hdl_memstream(fpgapart)

    def get_template_param_values(self):
        """Returns template parameter values for HLS code generation."""
        ret = dict()
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()

        # Fill in TSrcI
        ret["TSrcI"] = f"Slice<{inp_hls_str}>"
        # Fill in TDstI
        ret["TDstI"] = f"Slice<{out_hls_str}>"

        return ret

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing thresholds in appropriate format.

        Args:
            weights: numpy array with thresholds
            weight_file_mode: one of {hls_header, decoupled_npy,
                decoupled_verilog_dat, decoupled_runtime}
            weight_file_name: filename for the weight file
        """
        threshold_tensor = self.get_hw_compatible_threshold_tensor(weights)
        tdt = self.get_input_datatype(1)

        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), f"Thresholds can't be expressed with type {str(tdt)}"

        if weight_file_mode == "hls_header":
            # Save thresholds in thresh.h
            thresholds_hls_code = numpy_to_hls_code(
                threshold_tensor, tdt, "thresholds", False, True
            )

            # Write thresholds into thresh.h
            f_thresh = open(weight_file_name, "w")
            tdt_hls = tdt.get_hls_datatype_str()

            # Use binary to export bipolar activations
            export_odt = self.get_output_datatype()
            if export_odt == DataType["BIPOLAR"]:
                export_odt = DataType["BINARY"]
            odt_hls = export_odt.get_hls_datatype_str()

            f_thresh.write(
                f"static ThresholdsActivation<{self.calc_tmem()},{self.get_nodeattr('PE')},"
                f"{threshold_tensor.shape[-1]},{tdt_hls},{odt_hls},"
                f"{self.get_nodeattr('act_val')},comp::less_equal<{tdt_hls}, {tdt_hls}>> threshs = "
            )
            f_thresh.write(thresholds_hls_code)
            f_thresh.close()

        elif "decoupled" in weight_file_mode:
            # Streaming thresholds organized differently
            # (1, pe, tmem, n_thres_steps) -> (1, tmem, pe, n_thres_steps)
            decoupled_thres = np.transpose(threshold_tensor, (0, 2, 1, 3))

            pe = self.get_nodeattr("PE")
            n_thres_steps = self.get_nodeattr("num_steps")

            # Create PE-flipped version
            decoupled_thres_pe_flipped = np.flip(decoupled_thres, axis=-2)

            # Reshape to (1, tmem, pe * n_thres_steps)
            decoupled_thres = decoupled_thres.reshape(1, -1, pe * n_thres_steps)
            decoupled_thres = decoupled_thres.copy()

            decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.reshape(
                1, -1, pe * n_thres_steps
            )
            decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.copy()

            if weight_file_mode == "decoupled_npy":
                # Save weight stream into npy for cppsim
                np.save(weight_file_name, decoupled_thres)

            elif weight_file_mode == "decoupled_verilog_dat":
                # Convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                # Pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)

                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()

                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")

            elif weight_file_mode == "decoupled_runtime":
                # Memstream axi-lite interface maps each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_instream_width(1)
                words_per_memwidth = 2 ** ceil(log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32

                # Pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()

                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        # Split into groups of 8 hex digits (= 32 bits)
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise Exception("Decoupled weight export not yet implemented")
        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model, path):
        """Generate parameter files for HLS compilation."""
        code_gen_dir = path
        thresholds = model.get_initializer(self.onnx_node.input[1])
        mem_mode = self._get_mem_mode()

        if mem_mode == "embedded":
            # Save thresholds in thresh.h
            weight_filename = f"{code_gen_dir}/thresh.h"
            self.make_weight_file(thresholds, "hls_header", weight_filename)

        elif mem_mode == "decoupled":
            # Save decoupled weights for cppsim
            weight_filename_sim = f"{code_gen_dir}/thresholds.npy"
            self.make_weight_file(thresholds, "decoupled_npy", weight_filename_sim)

            # Also save weights as Verilog .dat file
            weight_filename_rtl = f"{code_gen_dir}/memblock.dat"
            self.make_weight_file(thresholds, "decoupled_verilog_dat", weight_filename_rtl)
        elif mem_mode == "dynamic":
            # Dynamic mode: thresholds streamed from external source (MLO)
            # No weight files needed - thresholds come from loop level
            pass
        else:
            raise Exception(f"Unrecognized mem_mode: {mem_mode}")

    def execute_node(self, context, graph):
        """Execute node in cppsim or rtlsim mode."""
        # Ensure design_space initialized (QONNX executor creates fresh instances)
        self._ensure_initialized_for_execution(graph)

        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        # Get code generation directory
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(f"Invalid exec_mode: {mode}. Must be 'cppsim' or 'rtlsim'")

        # Create npy file for each input
        in_ind = 0
        for inputs in node.input:
            if in_ind == 0:
                # Data input
                assert str(context[inputs].dtype) in [
                    "float32",
                    "float16",
                ], "Input datatype is not float32 or float16 as expected."

                expected_inp_shape = self.get_folded_input_shape()
                reshaped_input = context[inputs].reshape(expected_inp_shape)

                if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                    # Store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype(0)

                # Make copy before saving
                reshaped_input = reshaped_input.copy()
                np.save(os.path.join(code_gen_dir, f"input_{in_ind}.npy"), reshaped_input)

            elif in_ind > 2:
                raise Exception("Unexpected input found for Thresholding")

            in_ind += 1

        if mode == "cppsim":
            # Execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # Load output npy file
            super().npy_to_dynamic_output(context)

            # Reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out

            oshape = self.get_normal_output_shape()
            assert context[node.output[0]].shape == oshape, "Output shape is not as expected"

        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            super().reset_rtlsim(sim)

            mem_mode = self._get_mem_mode()
            # Decoupled and dynamic modes need threshold input
            if mem_mode in ("decoupled", "dynamic"):
                wnbits = self.get_instream_width(1)
                export_wdt = self.get_input_datatype(1)
                wei = npy_to_rtlsim_input(f"{code_gen_dir}/thresholds.npy", export_wdt, wnbits)
                num_w_reps = np.prod(self.get_nodeattr("num_input_vectors"))
                io_dict = {
                    "inputs": {"in0": inp, "in1": wei * num_w_reps},
                    "outputs": {"out0": []},
                }
            elif mem_mode == "embedded":
                io_dict = {
                    "inputs": {"in0": inp},
                    "outputs": {"out0": []},
                }
            else:
                raise Exception(f"Unrecognized mem_mode: {mem_mode}")

            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)

            output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = f"{code_gen_dir}/output_0.npy"
            out_shape = self.get_folded_output_shape()

            rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)

            # Load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output
        else:
            raise Exception(f"Invalid exec_mode: {mode}. Must be 'cppsim' or 'rtlsim'")

    def global_includes(self):
        """Generate global include directives."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']

        # Only embedded mode includes thresh.h header
        if self._get_mem_mode() == "embedded":
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var):
        """Generate HLS constant definitions."""
        num_reps = 1
        num_input_vectors = list(self.get_nodeattr("num_input_vectors"))
        total_spatial_size = int(np.prod(num_input_vectors))

        # Extract NumChannels from design_point (Arete principle)
        ki = self.design_point
        num_channels = ki.inputs["input"].tensor_shape[-1]

        self.code_gen_dict["$DEFINES$"] = [
            f"""#define NumChannels1 {num_channels}\n #define PE1 {self.get_nodeattr('PE')}\n #define numReps {num_reps}\n
               #define ImgDim1 {total_spatial_size}"""
        ]

        # Decoupled and dynamic modes need these defines for streaming interface
        if self._get_mem_mode() in ("decoupled", "dynamic"):
            self.code_gen_dict["$DEFINES$"].append(
                f"#define ActVal1 {self.get_nodeattr('act_val')}"
            )
            self.code_gen_dict["$DEFINES$"].append(
                f"#define ThresType1 {self.get_input_datatype(1).get_hls_datatype_str()}"
            )
            self.code_gen_dict["$DEFINES$"].append(
                f"#define NumSteps1 {self.get_nodeattr('num_steps')}"
            )

    def read_npy_data(self):
        """Generate code to read npy data for cppsim."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype(0)
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width(0)
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "half" if dtype == DataType["FLOAT16"] else "float"
        npy_in = f"{code_gen_dir}/input_0.npy"

        self.code_gen_dict["$READNPYDATA$"] = []

        # Note: innermost dim is reversed for input
        self.code_gen_dict["$READNPYDATA$"].append(
            f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("{npy_in}", in0_V, false);'
        )

        mem_mode = self._get_mem_mode()
        # Decoupled and dynamic modes need to read threshold data for cppsim
        if mem_mode in ("decoupled", "dynamic"):
            tdt = self.get_input_datatype(1)
            elem_bits = tdt.bitwidth()
            packed_bits = self.get_instream_width(1)
            packed_hls_type = f"ap_uint<{packed_bits}>"
            elem_hls_type = tdt.get_hls_datatype_str()
            npy_type = "half" if tdt == DataType["FLOAT16"] else "float"
            npy_in = f"{code_gen_dir}/thresholds.npy"

            self.code_gen_dict["$READNPYDATA$"].append(
                f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("{npy_in}", in1_V, false, ImgDim1);'
            )

    def strm_decl(self):
        """Generate stream declarations."""
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []

        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_instream_width(0)}>> in0_V ("in0_V");'
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");'
        )

        # Decoupled and dynamic modes have threshold streaming interface
        mem_mode = self._get_mem_mode()
        if mem_mode in ("decoupled", "dynamic"):
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{self.get_instream_width(1)}>> in1_V ("in1_V");'
            )

    def docompute(self):
        """Generate HLS docompute code."""
        tmpl_args = self.get_template_param_values()
        mem_mode = self._get_mem_mode()

        if mem_mode == "embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"""Thresholding_Batch<ImgDim1, NumChannels1, PE1, {tmpl_args['TSrcI']}, {tmpl_args['TDstI']}>
                (in0_V, out0_V, threshs, numReps);"""
            ]
        elif mem_mode in ("decoupled", "dynamic"):
            # Note: numReps is set to 1, repetition comes from threshold stream
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"""Thresholding_Stream_Batch<ImgDim1, NumChannels1, PE1, {tmpl_args['TSrcI']}, {tmpl_args['TDstI']}, ActVal1, ThresType1, NumSteps1>
                (in0_V, out0_V, in1_V, numReps);"""
            ]
        else:
            raise Exception(f"Unrecognized mem_mode: {mem_mode}")

    def dataoutstrm(self):
        """Generate code for output stream."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()

        if dtype == DataType["BIPOLAR"]:
            # Use binary for bipolar storage
            dtype = DataType["BINARY"]

        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = f"{code_gen_dir}/output_0.npy"
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # Note: innermost dim is not reversed for output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f'apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>(out0_V, {shape_cpp_str}, "{npy_out}", false);'
        ]

    def blackboxfunction(self):
        """Generate black box function signature."""
        mem_mode = self._get_mem_mode()

        if mem_mode == "embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"""void {self.onnx_node.name}(hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,
                    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V
                    )"""
            ]
        elif mem_mode in ("decoupled", "dynamic"):
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"""void {self.onnx_node.name}(hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,
                    hls::stream<ap_uint<{self.get_instream_width(1)}>> &in1_V,
                    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V
                    )"""
            ]
        else:
            raise Exception(f"Unrecognized mem_mode: {mem_mode}")

    def pragmas(self):
        """Generate HLS pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        mem_mode = self._get_mem_mode()
        if mem_mode == "embedded":
            # Threshold tensor is acc_type [PE][TMEM][N_THRES]
            # Partition for parallel access along PE and N_THRES dimensions (dims 1 and 3)
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3"
            )

            # Set resource type
            ram_style = self.get_nodeattr("ram_style")
            pe = self.get_nodeattr("PE")

            # Extract NumChannels from design_point (Arete principle)
            ki = self.design_point
            ich = ki.inputs["input"].tensor_shape[-1]

            # If PE < NumChannels, assign cores according to ram_style
            # Otherwise if PE == NumChannels, Vivado HLS will unroll to FFs
            if pe < ich:
                if ram_style == "distributed":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        "#pragma HLS RESOURCE variable=threshs.m_thresholds core=ROM_2P_LUTRAM"
                    )
                elif ram_style == "block":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        "#pragma HLS RESOURCE variable=threshs.m_thresholds core=ROM_2P_BRAM"
                    )
                else:
                    raise Exception(
                        f"Invalid ram_style: {ram_style}. Must be 'block' or 'distributed'"
                    )

        elif mem_mode in ("decoupled", "dynamic"):
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")

    def code_generation_ipi(self):
        """Generate TCL commands for IPI integration."""
        source_target = f"./ip/verilog/rtl_ops/{self.onnx_node.name}"
        cmd = [f"file mkdir {source_target}"]

        # Add streamer if needed (decoupled/dynamic modes)
        mem_mode = self._get_mem_mode()

        if mem_mode in ("decoupled", "dynamic"):
            node_name = self.onnx_node.name

            # Create hierarchy for this layer
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]

            cmd.append(f"create_bd_cell -type hier {node_name}")
            cmd.append(f"create_bd_pin -dir I -type clk /{node_name}/{clk_name}")
            cmd.append(f"create_bd_pin -dir I -type rst /{node_name}/{rst_name}")
            cmd.append(
                f"create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{dout_name}"
            )
            cmd.append(
                f"create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{din_name}"
            )

            # Instantiate HLS IP
            cmd.append(
                f"create_bd_cell -type ip -vlnv {self.get_nodeattr('ip_vlnv')} "
                f"/{node_name}/{node_name}"
            )

            # Instantiate streamer and connect to IP
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
            ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
            file_suffix = "_memstream_wrapper.v"

            # Find memstream verilog component in code generation directory
            strm_tmpl = None
            for fname in os.listdir(code_gen_dir):
                if fname.endswith(file_suffix):
                    strm_tmpl = fname
                    break

            if strm_tmpl is None:
                raise Exception("Memstream wrapper not found in code_gen_dir")

            strm_tmpl_name = strm_tmpl[:-2]

            sourcefiles = [
                os.path.join(code_gen_dir, strm_tmpl),
                axi_dir + "axilite.sv",
                ms_rtllib_dir + "memstream_axi.sv",
                ms_rtllib_dir + "memstream.sv",
            ]

            for f in sourcefiles:
                cmd += [f"add_files -copy_to {source_target} -norecurse {f}"]

            strm_inst = node_name + "_wstrm"
            cmd.append(
                f"create_bd_cell -type hier -reference {strm_tmpl_name} "
                f"/{node_name}/{strm_inst}"
            )

            # Connect streamer to HLS IP
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{strm_inst}/m_axis_0] "
                f"[get_bd_intf_pins {node_name}/{node_name}/in1_V]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{rst_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_rst_n]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_clk]"
            )

            # 2x clock not used for decoupled thresholds
            # Simply connect to 1x clock
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_clk2x]"
            )

            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{rst_name}] "
                f"[get_bd_pins {node_name}/{node_name}/{rst_name}]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/{node_name}/{clk_name}]"
            )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{din_name}] "
                f"[get_bd_intf_pins {node_name}/{node_name}/{din_name}]"
            )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{dout_name}] "
                f"[get_bd_intf_pins {node_name}/{node_name}/{dout_name}]"
            )

            # Note: AXI-lite runtime-writeable weights removed for simplicity
            cmd.append("save_bd_design")

        elif mem_mode == "embedded":
            # Base class impl sufficient for embedded mode
            return super().code_generation_ipi()

        else:
            raise Exception(f"Unrecognized mem_mode for Thresholding: {mem_mode}")

        return cmd

    def get_verilog_top_module_intf_names(self):
        """Get Verilog top module interface names."""
        intf_names = super().get_verilog_top_module_intf_names()

        # Note: AXI-lite support for runtime-writeable weights removed for simplicity
        # Decoupled and dynamic modes only expose streaming interfaces
        return intf_names

    def get_op_and_param_counts(self):
        """Get operation and parameter counts."""
        ret_dict = {}

        weight_bits = self.get_input_datatype(1).bitwidth()

        # Extract NumChannels from design_point (Arete principle)
        ki = self.design_point
        out_features = ki.inputs["input"].tensor_shape[-1]

        num_steps = self.get_nodeattr("num_steps")

        # Thresholds are called weights in this layer
        thres_param_type = f"param_threshold_{weight_bits}b"
        thres_count = out_features * num_steps
        ret_dict[thres_param_type] = thres_count

        return ret_dict

    def ipgen_extra_directives(self):
        """Return extra TCL directives for HLS synthesis."""
        return ["config_compile -pipeline_style frp"]

    def derive_characteristic_fxns(self, period):
        """Derive characteristic functions for performance estimation."""
        n_inps = np.prod(self.get_folded_input_shape()[:-1])

        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out0": []},
        }

        mem_mode = self._get_mem_mode()
        # Decoupled and dynamic modes have weight input
        if mem_mode in ("decoupled", "dynamic", "external"):
            n_weight_inps = self.calc_tmem()
            num_w_reps = np.prod(self.get_nodeattr("num_input_vectors"))
            io_dict["inputs"]["in1"] = [0 for i in range(num_w_reps * n_weight_inps)]

        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)
