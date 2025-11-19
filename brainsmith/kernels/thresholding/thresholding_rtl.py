############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
############################################################################

import math
import os
import shutil

import numpy as np
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_memutil_alternatives, mem_primitives_versal
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from brainsmith.kernels.thresholding.thresholding import Thresholding
from brainsmith.registry import backend


@backend(
    target_kernel="brainsmith:Thresholding",
    language="rtl",
    description="RTL implementation of Thresholding",
    author="AMD FINN"
)
class Thresholding_rtl(Thresholding, RTLBackend):
    """RTL backend for Thresholding kernel (KernelOp-based).

    This backend generates RTL code from finn-rtllib templates
    for the Thresholding operation.

    Key features:
    - Binary search-based threshold comparison
    - Resource estimation for BRAM/URAM/LUTRAM
    - Deep pipelining option for timing closure
    - Narrow-range quantization support
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Define nodeattrs for Thresholding_rtl backend.

        Combines:
        - Thresholding's schema-derived nodeattrs
        - RTLBackend's execution nodeattrs
        - RTL-specific parameters
        """
        my_attrs = {
            # Memory depth triggers for threshold storage
            "depth_trigger_uram": ("i", False, 0),
            "depth_trigger_bram": ("i", False, 0),
            # Enable uniform threshold optimization
            # (doesn't actually do anything yet, only for resource estimations)
            "uniform_thres": ("i", False, 0, {0, 1}),
            # Enable deep pipelining for easier timing closure
            # Setting to 0 may save FFs but otherwise leave on
            "deep_pipeline": ("i", False, 1, {0, 1}),
        }

        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))

        return my_attrs

    def get_pe_mem_geometries(self):
        """Return list of (bitwidth, depth) for PE memory configurations.

        Used in resource estimation. For each bitwidth, the depth is
        calculated as the number of thresholds that can be stored in
        a single memory block.

        Returns:
            List of (bitwidth, depth) tuples
        """
        pe = self.get_nodeattr("PE")
        wdt = self.get_input_datatype(1)
        wdt_bits = wdt.bitwidth()
        odt = self.get_output_datatype()
        odt_bits = odt.bitwidth()

        # Extract NumChannels from design_point block_shape (channels before stream tiling)
        ki = self.design_point
        t_channels = ki.inputs["input"].block_shape[-1]

        cf = t_channels / pe
        is_uniform = self.get_nodeattr("uniform_thres")

        if is_uniform:
            ret = [(odt_bits - x, cf * (2**x)) for x in range(1, odt_bits)]
        else:
            ret = [(wdt_bits, (cf) * 2**x) for x in range(odt_bits)]

        return ret

    def get_memory_estimate(self):
        """Return memory estimate for this node.

        Returns:
            Dictionary with resource type -> count
        """
        res_dict = {}
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        pe = self.get_nodeattr("PE")
        ret = self.get_pe_mem_geometries()

        for mem_cfg in ret:
            (width, depth) = mem_cfg
            primitives = mem_primitives_versal

            if depth_trigger_bram != 0 or depth_trigger_uram != 0:
                if depth >= depth_trigger_bram and depth < depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "BRAM" in k}
                elif depth >= depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "URAM" in k}

            alts = get_memutil_alternatives(mem_cfg, primitives)
            primary_alt = alts[0]
            res_type = primary_alt[0].split("_")[0]
            res_count, eff, waste = primary_alt[1]
            res_dict[res_type] = res_dict.get(res_type, 0) + pe * res_count

        return res_dict

    def bram_estimation(self):
        """Return number of BRAMs required for this node."""
        res_dict = self.get_memory_estimate()
        return res_dict.get("BRAM", 0)

    def uram_estimation(self):
        """Return number of URAMs required for this node."""
        res_dict = self.get_memory_estimate()
        return res_dict.get("URAM", 0)

    def lut_estimation(self):
        """Return number of LUTs required for this node."""
        res_dict = self.get_memory_estimate()
        return res_dict.get("LUTRAM", 0)

    def get_all_meminit_filenames(self, abspath=False):
        """Return list of all .dat memory initializer files.

        Args:
            abspath: If True, return absolute paths; otherwise relative

        Returns:
            List of .dat filenames
        """
        dat_files = []
        t_path = self.get_nodeattr("code_gen_dir_ipgen") if abspath else "."
        pe = self.get_nodeattr("PE")
        odt = self.get_output_datatype(0)
        o_bitwidth = odt.bitwidth()

        for stage in range(o_bitwidth):
            for pe_value in range(pe):
                thresh_file = f"{t_path}/{self.onnx_node.name}_threshs_{pe_value}_{stage}.dat"
                dat_files.append(thresh_file)

        return dat_files

    def prepare_codegen_rtl_values(self, model):
        """Prepare dictionary values for RTL template substitution.

        Returns:
            Dictionary mapping template variables to values
        """
        code_gen_dict = {}
        t_path = self.get_nodeattr("code_gen_dir_ipgen")

        self.generate_params(model, t_path)

        bias = self.get_nodeattr("act_val")
        odt = self.get_output_datatype(0)
        idt = self.get_input_datatype(0)
        o_bitwidth = odt.bitwidth()
        pe = self.get_nodeattr("PE")

        # Extract NumChannels from design_point block_shape (channels before stream tiling)
        ki = self.design_point
        num_channels = ki.inputs["input"].block_shape[-1]

        # RTL expects 2^N-1 thresholds, but narrow range quantization results in
        # one less threshold. Prepend a dummy threshold (minimal possible value
        # determined by input datatype) and decrease bias by 1.
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.get_nodeattr("num_steps")
        wdt = self.get_input_datatype(1)

        if expected_thresholds != n_thres_steps:
            if odt.signed():
                bias = bias - 1
            else:
                max_val = wdt.max()
                if max_val <= idt.max():
                    max_val = max_val + 1
                    # Increase wdt
                    if not wdt.signed():
                        wdt = DataType.get_smallest_possible(max_val)
                    else:
                        wdt = DataType.get_smallest_possible(-max_val - 1)

        code_gen_dict["$THRESHOLDS_PATH$"] = [f'"./{self.onnx_node.name}_"']

        # Identify module name
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]

        # Set top module name - AXI wrapper
        code_gen_dict["$TOP_MODULE$"] = code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"]

        # Identify module variables
        i_bitwidth = idt.bitwidth()

        code_gen_dict["$N$"] = [str(2**o_bitwidth - 1)]  # Number of needed thresholds
        code_gen_dict["$WT$"] = [str(wdt.bitwidth())]  # Threshold precision
        code_gen_dict["$WI$"] = [str(i_bitwidth)]  # Input precision
        code_gen_dict["$C$"] = [str(num_channels)]  # Number of channels
        code_gen_dict["$BIAS$"] = [str(bias)]  # Activation bias value
        code_gen_dict["$PE$"] = [str(pe)]  # PE

        # Is input datatype signed or unsigned?
        # Thresholding core needs to know this when comparing weights to inputs
        if self.get_input_datatype(0).signed():
            code_gen_dict["$SIGNED$"] = [str(1)]
        else:
            code_gen_dict["$SIGNED$"] = [str(0)]

        # Is input datatype non-integer? (assume this means floating-point)
        if self.get_input_datatype().is_integer():
            code_gen_dict["$FPARG$"] = [str(0)]
        else:
            code_gen_dict["$FPARG$"] = [str(1)]

        # Calculate output bits
        if bias >= 0:
            o_bits = math.ceil(math.log2(2**o_bitwidth + bias))
        else:
            o_bits = 1 + math.ceil(
                math.log2(-bias if -bias >= 2 ** (o_bitwidth - 1) else 2**o_bitwidth + bias)
            )
        code_gen_dict["$O_BITS$"] = [str(int(o_bits))]

        # Runtime-writable weights (AXI-lite support removed)
        code_gen_dict["$USE_AXILITE$"] = ["0"]

        # Depth triggers and deep pipeline
        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        deep_pipeline = self.get_nodeattr("deep_pipeline")

        code_gen_dict["$DEPTH_TRIGGER_URAM$"] = [str(depth_trigger_uram)]
        code_gen_dict["$DEPTH_TRIGGER_BRAM$"] = [str(depth_trigger_bram)]
        code_gen_dict["$DEEP_PIPELINE$"] = [str(deep_pipeline)]

        return code_gen_dict

    def get_rtl_file_list(self, abspath=False):
        """Return list of RTL source files.

        Args:
            abspath: If True, return absolute paths; otherwise relative

        Returns:
            List of RTL filenames
        """
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/thresholding/hdl/")
            axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""
            axi_dir = ""

        verilog_files = [
            axi_dir + "axilite.sv",
            rtllib_dir + "thresholding.sv",
            rtllib_dir + "thresholding_axi.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]

        return verilog_files

    def generate_hdl(self, model, fpgapart, clk):
        """Prepare HDL files from templates for synthesis.

        Args:
            model: Model wrapper
            fpgapart: FPGA part name
            clk: Clock period
        """
        # Generate dictionary of values to put in RTL template
        code_gen_dict = self.prepare_codegen_rtl_values(model)

        # Retrieve destination directory for final RTL files
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        # Set 'gen_top_module' attribute for use later by xsi and IPI generation
        self.set_nodeattr("gen_top_module", code_gen_dict["$TOP_MODULE$"][0])

        axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/thresholding/hdl"
        template_path = rtlsrc + "/thresholding_template_wrapper.v"

        with open(template_path) as f:
            template_wrapper = f.read()

        # Replace template variables
        for key in code_gen_dict:
            # Transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template_wrapper = template_wrapper.replace(key, code_gen_line)

        with open(os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v"), "w") as f:
            f.write(template_wrapper)

        # Copy RTL source files
        sv_files = ["thresholding.sv", "thresholding_axi.sv"]
        for sv_file in sv_files:
            shutil.copy(rtlsrc + "/" + sv_file, code_gen_dir)
        shutil.copy(axi_dir + "axilite.sv", code_gen_dir)

        # Set ipgen_path and ip_path so that HLS-Synth transformation
        # and stitch_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def execute_node(self, context, graph):
        """Execute node in cppsim or rtlsim mode."""
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        if mode == "cppsim":
            # Use parent class cppsim execution
            Thresholding.execute_node(self, context, graph)

        elif mode == "rtlsim":
            node = self.onnx_node

            # Create npy file for each input
            in_ind = 0
            for inputs in node.input:
                # First input is data, second is thresholds
                if in_ind == 0:
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
                    raise Exception("Unexpected input found for Thresholding_rtl")

                in_ind += 1

            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)

            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }

            super().reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)

            rtlsim_output = io_dict["outputs"]["out0"]

            # Manage output data
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = f"{code_gen_dir}/output.npy"
            out_shape = self.get_folded_output_shape()

            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            # Load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output

        else:
            raise Exception(f"Invalid exec_mode: {mode}. Must be 'cppsim' or 'rtlsim'")

    def code_generation_ipi(self):
        """Constructs and returns TCL commands for node instantiation as RTL block."""
        rtl_file_list = self.get_rtl_file_list()
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        source_target = f"./ip/verilog/rtl_ops/{self.onnx_node.name}"
        cmd = [f"file mkdir {source_target}"]

        for rtl_file in rtl_file_list:
            cmd.append(
                f"add_files -copy_to {source_target} -norecurse "
                f"{os.path.join(code_gen_dir, rtl_file)}"
            )

        # Create RTL block, not an IP core (-type ip)
        cmd.append(
            f"create_bd_cell -type module -reference {self.get_nodeattr('gen_top_module')} "
            f"{self.onnx_node.name}"
        )

        return cmd

    def get_verilog_top_module_intf_names(self):
        """Get Verilog top module interface names."""
        return super().get_verilog_top_module_intf_names()

    def generate_params(self, model, path):
        """Generate parameter files for RTL compilation.

        Args:
            model: Model wrapper
            path: Output directory path
        """
        thresholds = model.get_initializer(self.onnx_node.input[1])
        # Skip if no initializer (will be provided at runtime)
        if thresholds is None:
            return
        file_name = f"{path}/memblock.dat"
        self.make_weight_file(thresholds, "internal_embedded", file_name)

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce file containing thresholds in appropriate format.

        Args:
            weights: numpy array with thresholds
            weight_file_mode: one of {decoupled_runtime, internal_embedded}
            weight_file_name: filename for weight file
        """
        path = os.path.dirname(weight_file_name)
        if not path:
            path = os.getcwd()

        thresholds = weights
        pe = self.get_nodeattr("PE")

        # Extract NumChannels from design_point block_shape (channels before stream tiling)
        ki = self.design_point
        num_channels = ki.inputs["input"].block_shape[-1]

        odt = self.get_output_datatype(0)
        idt = self.get_input_datatype(0)
        o_bitwidth = odt.bitwidth()

        # RTL expects 2^N-1 thresholds, but narrow range quantization results in
        # one less threshold. Prepend/append dummy threshold and increase numSteps.
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.get_nodeattr("num_steps")
        wdt = self.get_input_datatype(1)

        if expected_thresholds != n_thres_steps:
            if odt.signed():
                min_val = wdt.min()
                thresholds = np.insert(thresholds, 0, min_val, axis=1)
            else:
                # Temporary fix for unsigned narrow quantization
                max_val = wdt.max()
                if max_val > idt.max():
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
                else:
                    max_val = max_val + 1
                    # Increase wdt
                    if not wdt.signed():
                        wdt = DataType.get_smallest_possible(max_val)
                    else:
                        wdt = DataType.get_smallest_possible(-max_val - 1)
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)

            n_thres_steps += 1

        if weight_file_mode == "decoupled_runtime":
            # If single threshold value found, tile to all channels (per-tensor quantization)
            if thresholds.shape[0] == 1:
                thresholds = np.tile(thresholds, (num_channels, 1))

            width_padded = roundup_to_integer_multiple(thresholds.shape[1], 2**o_bitwidth)
            thresh_padded = np.zeros((thresholds.shape[0], width_padded))
            thresh_padded[: thresholds.shape[0], :n_thres_steps] = thresholds
            thresh_stream = []
            bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 32)
            padding = np.zeros(width_padded, dtype=np.int32)

            chan_ind = 0
            cf = num_channels // pe

            for fold in range(cf):
                for c in range(2 ** (pe - 1).bit_length()):
                    if (c == 0 or c % pe != 0) and c < pe:
                        for t in thresh_padded[chan_ind]:
                            t_packed = pack_innermost_dim_as_hex_string(
                                [t], wdt, bw_hexdigit, prefix=""
                            ).item()
                            thresh_stream.append(t_packed)
                        chan_ind += 1
                    else:
                        for z in padding:
                            t_packed = pack_innermost_dim_as_hex_string(
                                [z], wdt, bw_hexdigit, prefix=""
                            ).item()
                            thresh_stream.append(t_packed)

            with open(weight_file_name, "w") as f:
                for val in thresh_stream:
                    f.write(val + "\n")

        elif weight_file_mode == "internal_embedded":
            # If single threshold value found, tile to all channels (per-tensor quantization)
            if thresholds.shape[0] == 1:
                thresholds = np.tile(thresholds, (num_channels, 1))

            # Add dummy dimension as final dimension (gets packed)
            t_expand = np.expand_dims(thresholds, axis=-1)
            bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
            t_packed = pack_innermost_dim_as_hex_string(t_expand, wdt, bw_hexdigit, prefix="")

            channel_fold = int(num_channels / pe)

            for stage in range(o_bitwidth):
                sn = o_bitwidth - stage - 1
                for pe_value in range(pe):
                    thresh_file = f"{path}/{self.onnx_node.name}_threshs_{pe_value}_{stage}.dat"
                    threshs = np.zeros([channel_fold * (2**stage)], dtype="object")

                    for ch in range(channel_fold):
                        for i in range(2**stage):
                            threshs[(ch << stage) + i] = t_packed[ch * pe + pe_value][
                                (i << (o_bitwidth - stage)) + 2**sn - 1
                            ]

                    with open(thresh_file, "w") as f:
                        for val in threshs:
                            f.write(val + "\n")
