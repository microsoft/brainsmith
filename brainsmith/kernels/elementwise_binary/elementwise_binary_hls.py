# Portions derived from FINN project
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HLS backend for ElementwiseBinaryOp (KernelOp-based).

Implements hardware synthesis and C++ simulation for elementwise binary operations.

Phase 1 Scope:
- Dynamic LHS (streaming input) + Static RHS (parameter) pattern only
- Simplified from FINN's complex broadcast logic
- Leverages KernelOp design_point system for shape management
- Supports 17 binary operations via polymorphic cpp_op

Phase 2/3 (Future):
- Dynamic + Dynamic inputs with broadcast
- internal_decoupled memory mode
- Layout optimizations
"""

import logging
from dataclasses import dataclass
from math import ceil

import numpy as np
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import (
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
)
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from brainsmith.kernels.elementwise_binary.elementwise_binary import ElementwiseBinaryOp
from brainsmith.registry import backend

logger = logging.getLogger(__name__)


@dataclass
class BufferDeclaration:
    """HLS buffer declaration with partition pragma.

    Represents a buffer array declaration and its associated HLS pragma
    for array partitioning. Provides type-safe alternative to tuple returns.

    Attributes:
        declaration: C++ array declaration (e.g., "LhsType lhs[1][64][2];")
        partition_pragma: HLS pragma for array partitioning
    """

    declaration: str
    partition_pragma: str

    def emit(self) -> list[str]:
        """Emit buffer declaration as code lines.

        Returns:
            List containing declaration and pragma as separate lines
        """
        return [self.declaration, self.partition_pragma]


@backend(
    target_kernel="brainsmith:ElementwiseBinaryOp",
    language="hls",
    author="AMD FINN",
)
class ElementwiseBinaryOp_hls(ElementwiseBinaryOp, HLSBackend):
    """HLS backend for ElementwiseBinaryOp (KernelOp-based).

    Supports HLS code generation and C++ simulation for 17 binary operations:
    - Arithmetic: Add, Sub, Mul, Div
    - Logical: And, Or, Xor
    - Comparison: Equal, Less, LessOrEqual, Greater, GreaterOrEqual
    - Bitwise: BitwiseAnd, BitwiseOr, BitwiseXor
    - BitShift: BitShiftLeft, BitShiftRight

    Phase 1: LHS streaming + RHS static parameter pattern only.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Combine ElementwiseBinaryOp and HLSBackend nodeattrs."""
        my_attrs = ElementwiseBinaryOp.get_nodeattr_types(self)
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        # Add python exec_mode support (Phase 3b)
        my_attrs["exec_mode"] = ("s", False, "", {"", "python", "cppsim", "rtlsim"})
        return my_attrs

    # ================================================================
    # Resource Estimation (Uses design_point)
    # ================================================================

    def _calculate_bram_usage(self) -> int:
        """Calculate BRAM usage (extracted for clarity and testing).

        Returns:
            Number of BRAM blocks required
        """
        style = self.get_nodeattr("ram_style")
        lhs_iface = self.design_point.inputs["lhs"]
        pe = lhs_iface.stream_shape[-1]  # PE from stream tiling
        rhs_dtype = self.get_input_datatype(1)
        A = rhs_dtype.bitwidth()

        # Calculate memory size for RHS parameter
        rhs_tensor_shape = self.design_point.inputs["rhs"].tensor_shape
        tmem = int(np.prod(rhs_tensor_shape[:-1]))  # All dims except PE

        if style == "block" and tmem > 1:
            return int(ceil(A * pe / 16)) * int(ceil(tmem / 1024))
        else:
            return 0

    def bram_estimation(self):
        """Calculates BRAM cost with validation against target device capacity.

        Returns:
            Number of BRAM blocks required

        Warnings:
            Logs warning if estimated usage exceeds device capacity
            Logs info if using >80% of device capacity
        """
        bram_blocks = self._calculate_bram_usage()

        # Validate against target device (if known)
        target_device = getattr(self, "target_device", None)
        if target_device is not None and hasattr(target_device, "bram_count"):
            max_bram = target_device.bram_count
            if bram_blocks > max_bram:
                logger.warning(
                    f"{self.onnx_node.name}: BRAM estimate ({bram_blocks}) "
                    f"exceeds device capacity ({max_bram}). "
                    f"Consider reducing PE or using distributed RAM (ram_style='distributed')."
                )
            elif bram_blocks > max_bram * 0.8 and bram_blocks > 0:
                logger.info(
                    f"{self.onnx_node.name}: BRAM estimate ({bram_blocks}) "
                    f"uses {bram_blocks/max_bram:.0%} of device capacity."
                )

        return bram_blocks

    def _calculate_lut_usage(self) -> int:
        """Calculate LUT usage (extracted for clarity and testing).

        Returns:
            Number of LUTs required
        """
        style = self.get_nodeattr("ram_style")
        lhs_iface = self.design_point.inputs["lhs"]
        pe = lhs_iface.stream_shape[-1]  # PE from stream tiling
        lhs_dtype = self.get_input_datatype(0)
        rhs_dtype = self.get_input_datatype(1)

        # Operation width depends on larger input
        A = max(lhs_dtype.bitwidth(), rhs_dtype.bitwidth())

        # Calculate memory size for RHS parameter
        rhs_tensor_shape = self.design_point.inputs["rhs"].tensor_shape
        tmem = int(np.prod(rhs_tensor_shape[:-1]))

        # Cost of operations/comparators (PE-parallel)
        operation_cost = A * pe

        # Cost of LUTRAM for parameters
        if style == "distributed" and tmem > 1:
            lutram_cost = pe * rhs_dtype.bitwidth() * int(ceil(tmem / 64))
        else:
            lutram_cost = 0

        return operation_cost + lutram_cost

    def lut_estimation(self):
        """Calculates LUT cost with validation against target device capacity.

        Returns:
            Number of LUTs required

        Warnings:
            Logs warning if estimated usage exceeds device capacity
            Logs info if using >80% of device capacity
        """
        lut_count = self._calculate_lut_usage()

        # Validate against target device (if known)
        target_device = getattr(self, "target_device", None)
        if target_device is not None and hasattr(target_device, "lut_count"):
            max_luts = target_device.lut_count
            if lut_count > max_luts:
                logger.warning(
                    f"{self.onnx_node.name}: LUT estimate ({lut_count}) "
                    f"exceeds device capacity ({max_luts}). "
                    f"Consider reducing PE or operation bitwidth."
                )
            elif lut_count > max_luts * 0.8 and lut_count > 0:
                logger.info(
                    f"{self.onnx_node.name}: LUT estimate ({lut_count}) "
                    f"uses {lut_count/max_luts:.0%} of device capacity."
                )

        return lut_count

    def dsp_estimation(self, fpgapart):
        """DSP usage - only Mul might use DSP blocks."""
        func = self.get_nodeattr("func")
        if func == "Mul":
            # Mul could use DSP, but depends on synthesis settings
            # Conservative estimate: 0 (assumes LUT-based multiply)
            return 0
        return 0

    # ================================================================
    # Broadcasting Helper Methods (Phase 2)
    # ================================================================

    def _get_broadcast_info(self, input_name):
        """Get BroadcastInfo for broadcasting analysis (cached).

        Computes broadcasting metadata between LHS and RHS input shapes.
        Returns None if neither input is streaming (no broadcasting possible).
        Returns the same BroadcastInfo for both "lhs" and "rhs" queries.

        Args:
            input_name: Name of input ("lhs" or "rhs") - used for validation only

        Returns:
            BroadcastInfo object, or None if no streaming inputs
        """
        from brainsmith.dataflow.broadcast_helpers import BroadcastInfo

        # Cache broadcast info once for both inputs (it's the same object)
        cache_attr = "_broadcast_info"
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        # Check if at least one input is streaming
        lhs_streaming = self._needs_streaming_interface("lhs")
        rhs_streaming = self._needs_streaming_interface("rhs")

        if not lhs_streaming and not rhs_streaming:
            # No streaming inputs → no broadcasting
            setattr(self, cache_attr, None)
            return None

        # Get shapes from design_point
        lhs_shape = self.design_point.inputs["lhs"].tensor_shape
        rhs_shape = self.design_point.inputs["rhs"].tensor_shape

        # Compute broadcast info between LHS and RHS
        broadcast_info = BroadcastInfo.compute(lhs_shape, rhs_shape)

        # Cache and return
        setattr(self, cache_attr, broadcast_info)
        return broadcast_info

    def _needs_streaming_interface(self, input_name):
        """Check if input needs a streaming (dynamic) interface.

        Determines streaming vs static based on mem_mode (single source of truth):
        - RHS with mem_mode in ("embedded", "decoupled"): static (loaded from params.hpp)
        - RHS with mem_mode == "dynamic": streaming (MLO case)
        - RHS with no mem_mode: streaming (not a weight)
        - LHS: always streaming

        Args:
            input_name: "lhs" or "rhs"

        Returns:
            True if input should be streamed, False if static parameter
        """
        if input_name == "lhs":
            # LHS is always streaming
            return True

        # RHS: check mem_mode to determine if static or streaming
        rhs_iface = self.design_point.inputs.get("rhs")
        if rhs_iface and hasattr(rhs_iface, "mem_mode") and rhs_iface.mem_mode:
            # Weight with mem_mode set
            if rhs_iface.mem_mode in ("embedded", "decoupled"):
                return False  # Static - loaded from params.hpp
            else:  # "dynamic"
                return True  # Streaming - MLO case

        # No mem_mode or not a weight - streaming
        return True

    def _get_buffer_declaration(self, input_name: str, pe: int) -> BufferDeclaration | None:
        """Generate buffer array declaration for an input.

        Args:
            input_name: "lhs" or "rhs"
            pe: PE parallelism factor

        Returns:
            BufferDeclaration with declaration and pragma, or None for static inputs
        """
        if not self._needs_streaming_interface(input_name):
            # Static inputs don't need runtime buffers (loaded from params.hpp)
            return None

        broadcast_info = self._get_broadcast_info(input_name)
        if not broadcast_info or not broadcast_info.has_broadcast:
            # No broadcasting - simple buffer
            output_shape = self.design_point.outputs["output"].tensor_shape
            # Divide last dimension by PE (folded shape)
            buffer_shape = output_shape[:-1] + (output_shape[-1] // pe,)
        else:
            # Broadcasting - use BroadcastInfo to compute buffer shape
            buffer_shape = broadcast_info.get_buffer_shape(input_name, pe)

        # Generate type based on input
        if input_name == "lhs":
            dtype_str = "LhsType"
        else:  # rhs
            dtype_str = "RhsType"

        # Generate multi-dimensional array declaration
        # Example: LhsType lhs[1][64][64][2];
        dim_str = "".join(f"[{dim}]" for dim in buffer_shape)
        declaration = f"{dtype_str} {input_name}{dim_str};"

        # Generate partition pragma for last dimension (PE parallelism)
        ndim = len(buffer_shape)
        partition_pragma = f"#pragma HLS ARRAY_PARTITION variable={input_name} complete dim={ndim}"

        return BufferDeclaration(declaration=declaration, partition_pragma=partition_pragma)

    def _get_read_condition(self, input_name, loop_counters):
        """Generate C++ condition for when to read from input stream.

        Args:
            input_name: "lhs" or "rhs"
            loop_counters: Tuple of loop counter variable names (e.g., ("i0", "i1", "i2"))

        Returns:
            C++ boolean expression string, or "true" if always read
        """
        broadcast_info = self._get_broadcast_info(input_name)
        if not broadcast_info or not broadcast_info.has_broadcast:
            # No broadcasting - always read
            return "true"

        # Use BroadcastInfo to generate read condition
        return broadcast_info.should_read_new_value(input_name, loop_counters) or "true"

    def _get_indexing_expression(self, input_name, loop_counters, pe_var="pe"):
        """Generate C++ array indexing expression for buffer access.

        Args:
            input_name: "lhs" or "rhs"
            loop_counters: Tuple of loop counter variable names
            pe_var: PE loop variable name (default "pe")

        Returns:
            C++ indexing expression string (e.g., "[i0][i1][i2][pe]")
        """
        broadcast_info = self._get_broadcast_info(input_name)
        if not broadcast_info or not broadcast_info.has_broadcast:
            # No broadcasting - use all counters + PE
            output_shape = self.design_point.outputs["output"].tensor_shape
            ndim = len(output_shape) - 1  # Exclude PE dimension
            indices = list(loop_counters[:ndim]) + [pe_var]
            return "".join(f"[{idx}]" for idx in indices)

        # Use BroadcastInfo to generate broadcast-aware indexing
        return broadcast_info.get_index_expression(input_name, loop_counters, pe_var)

    # ================================================================
    # HLS Code Generation
    # ================================================================

    def get_template_param_values(self):
        """Returns template parameter values for HLS code."""
        ret = dict()
        lhs_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        rhs_hls_str = self.get_input_datatype(1).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        ret["LhsType"] = lhs_hls_str
        ret["RhsType"] = rhs_hls_str
        ret["OutType"] = out_hls_str
        # Slice template uses default construction with curly braces, not function-style cast
        # Pattern from FINN: Slice<LhsType>{}(packed_value)
        ret["LhsSlice"] = f"Slice<{lhs_hls_str}>{{}}"
        ret["RhsSlice"] = f"Slice<{rhs_hls_str}>{{}}"
        ret["OutSlice"] = f"Slice<{out_hls_str}>{{}}"
        return ret

    def generate_params(self, model, path):
        """Generate params.hpp with static parameter tensors.

        For dynamic_static pattern: Generates RHS parameter array
        For dynamic_dynamic pattern: Creates empty params.hpp (no static inputs)
        For MLO (dynamic mem_mode): Generates memblock.dat for FINNLoop

        Implements FINN-compatible parameter reshaping:
        1. Reshape to folded input shape (matches PE-parallelized access)
        2. Broadcast to PE dimension if needed
        3. Pad dimensions from left to align with output shape for broadcasting
        """
        code_gen_dir = path

        # Collect parameter code for static inputs
        param_code_sections = []

        # Initialize pragmas list if not already exists (needed for parameter pragmas)
        if "$PRAGMAS$" not in self.code_gen_dict:
            self.code_gen_dict["$PRAGMAS$"] = []

        # Folded output shape for broadcasting/aligning the input shapes
        out_shape = self.get_folded_output_shape(ind=0)
        pe = self.get_nodeattr("PE")

        # Check LHS (only static in rare cases, but handle for completeness)
        if not self._needs_streaming_interface("lhs"):
            lhs_parameters = model.get_initializer(self.onnx_node.input[0])
            if lhs_parameters is not None:
                lhs_dtype = DataType[self.get_input_datatype(0).name]

                # FINN-compatible reshaping: folded shape → PE broadcast → dimension padding
                lhs_parameters = lhs_parameters.reshape(*self.get_folded_input_shape(ind=0))

                # Broadcast to PE dimension if needed
                if lhs_parameters.shape[-1] != pe:
                    lhs_parameters = np.broadcast_to(
                        lhs_parameters, lhs_parameters.shape[:-1] + (pe,)
                    )

                # Pad dimensions from left to align with output shape
                lhs_shape = lhs_parameters.shape
                lhs_shape = (len(out_shape) - len(lhs_shape)) * (1,) + lhs_shape
                lhs_parameters = lhs_parameters.reshape(*lhs_shape)

                lhs_code = numpy_to_hls_code(lhs_parameters, lhs_dtype, "lhs", False, False)

                param_code_sections.append("// LHS parameter tensor\n")
                param_code_sections.append(lhs_code)

                # Add HLS pragmas for parameter storage and partitioning
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS BIND_STORAGE variable=lhs type=ROM_2P impl=distributed"
                )
                self.code_gen_dict["$PRAGMAS$"].append(
                    f"#pragma HLS ARRAY_PARTITION variable=lhs complete dim={len(lhs_shape)}"
                )

        # Check RHS - handle both static (embedded) and MLO (dynamic) cases
        # For MLO, RHS is streaming but FINNLoop.generate_params() sets the initializer
        rhs_parameters = model.get_initializer(self.onnx_node.input[1])

        # Determine if this is MLO mode (dynamic mem_mode with initializer)
        rhs_iface = self.design_point.inputs.get("rhs")
        is_mlo_mode = (
            rhs_iface is not None
            and hasattr(rhs_iface, "mem_mode")
            and rhs_iface.mem_mode == "dynamic"
        )

        if rhs_parameters is not None:
            rhs_dtype = DataType[self.get_input_datatype(1).name]

            # FINN-compatible reshaping: folded shape → PE broadcast → dimension padding
            rhs_parameters = rhs_parameters.reshape(*self.get_folded_input_shape(ind=1))

            # Broadcast to PE dimension if needed
            if rhs_parameters.shape[-1] != pe:
                rhs_parameters = np.broadcast_to(rhs_parameters, rhs_parameters.shape[:-1] + (pe,))

            # Pad dimensions from left to align with output shape
            rhs_shape = rhs_parameters.shape
            rhs_shape = (len(out_shape) - len(rhs_shape)) * (1,) + rhs_shape
            rhs_parameters = rhs_parameters.reshape(*rhs_shape)

            if is_mlo_mode:
                # MLO mode: write memblock.dat for FINNLoop.generate_params()
                # This matches FINN's ElementwiseBinaryOperation_hls behavior
                # Merge first dimensions together for streaming format
                rhs_flat = rhs_parameters.reshape(-1, pe)
                # Flip PE dimension (FINN convention for streaming)
                rhs_flat = np.flip(rhs_flat, axis=-1)
                rhs_width = self.get_instream_width(1)
                # Pad to nearest 4 bits to get hex strings
                rhs_width_padded = roundup_to_integer_multiple(rhs_width, 4)
                rhs_tensor = pack_innermost_dim_as_hex_string(
                    rhs_flat, rhs_dtype, rhs_width_padded, prefix=""
                )
                rhs_stream = rhs_tensor.flatten()
                rhs_stream = rhs_stream.copy()
                with open(f"{code_gen_dir}/memblock.dat", "w") as f:
                    for val in rhs_stream:
                        f.write(val + "\n")
            elif not self._needs_streaming_interface("rhs"):
                # Static embedded mode: write to params.hpp
                rhs_code = numpy_to_hls_code(rhs_parameters, rhs_dtype, "rhs", False, False)

                param_code_sections.append("// RHS parameter tensor\n")
                param_code_sections.append(rhs_code)

                # Add HLS pragmas for parameter storage and partitioning
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS BIND_STORAGE variable=rhs type=ROM_2P impl=distributed"
                )
                self.code_gen_dict["$PRAGMAS$"].append(
                    f"#pragma HLS ARRAY_PARTITION variable=rhs complete dim={len(rhs_shape)}"
                )
        elif not self._needs_streaming_interface("rhs"):
            # Static mode but no initializer - error
            raise ValueError(
                f"ElementwiseBinaryOp with static RHS (mem_mode=embedded/decoupled) requires RHS parameter, "
                f"but {self.onnx_node.input[1]} is not an initializer"
            )

        # Write params.hpp
        with open(f"{code_gen_dir}/params.hpp", "w") as f:
            if param_code_sections:
                f.write("".join(param_code_sections))
            else:
                # No static parameters (dynamic_dynamic pattern or MLO)
                f.write("// No static parameters (inputs are streaming or MLO)\n")

    def execute_node(self, context, graph):
        """Execute ElementwiseBinaryOp in python, cppsim, or rtlsim mode.

        Args:
            context: Execution context containing input/output tensors
            graph: ONNX graph

        Modes:
        - python: Pure Python/numpy execution (golden reference)
        - cppsim: C++ simulation (requires HLS compilation) - uses HLSBackend
        - rtlsim: RTL simulation (requires synthesis) - uses HLSBackend

        Raises:
            ValueError: If exec_mode is not set or invalid
            RuntimeError: If compilation fails (cppsim/rtlsim)
            Exception: If VITIS_PATH/HLS_PATH not set (cppsim/rtlsim)

        Environment Requirements (cppsim/rtlsim):
            - VITIS_PATH or HLS_PATH environment variable must be set
            - Environment must be sourced BEFORE running Python:
                source .brainsmith/env.sh
              or:
                direnv allow
            - Compilation may take 2-10 minutes on first run
            - Subsequent runs use cached executable

        Example:
            # Ensure environment is sourced before running:
            #   $ source .brainsmith/env.sh
            #
            >>> hw_op.set_nodeattr("exec_mode", "cppsim")
            >>> result = hw_op.execute_node(context, model.graph)
        """
        # Ensure initialized before mode dispatch
        self._ensure_initialized_for_execution(graph)

        mode = self.get_nodeattr("exec_mode")

        if mode == "python":
            # Custom Python execution for golden reference
            self._execute_python(context, graph)
        elif mode in ["cppsim", "rtlsim"]:
            # Delegate to HLSBackend's execute_node for cppsim/rtlsim
            # It handles all the C++ compilation, input/output prep, and execution
            super().execute_node(context, graph)
        else:
            raise ValueError(
                f"Invalid or unset exec_mode: '{mode}'. " f"Must be 'python', 'cppsim', or 'rtlsim'"
            )

    def _execute_python(self, context, graph):
        """Execute in pure Python mode using numpy operations.

        This provides a golden reference implementation that matches hardware
        behavior but runs entirely in Python without HLS compilation.

        Args:
            context: Execution context dict (tensor_name -> numpy array)
            graph: ONNX graph (for metadata)
        """
        node = self.onnx_node

        # Get inputs from context
        lhs = context[node.input[0]]
        rhs = context[node.input[1]]

        # Convert to int64 for integer computation (avoid overflow)
        lhs_dtype = self.design_point.inputs["lhs"].datatype
        rhs_dtype = self.design_point.inputs["rhs"].datatype

        if lhs_dtype.is_integer():
            lhs = lhs.astype(np.int64)
        if rhs_dtype.is_integer():
            rhs = rhs.astype(np.int64)

        # Apply operation with numpy broadcasting
        result = self.npy_op(lhs, rhs)

        # Apply output quantization
        output_dtype = self.design_point.outputs["output"].datatype
        if output_dtype.is_integer():
            result = self._quantize_output(result, output_dtype)

        # Store result (QONNX convention: store in float32 container)
        context[node.output[0]] = result.astype(np.float32)

    def _quantize_output(self, values, datatype):
        """Quantize output values to match hardware datatype.

        Args:
            values: Numpy array of values to quantize
            datatype: Target DataType for quantization

        Returns:
            Quantized numpy array clipped to datatype range
        """
        # Get datatype range
        min_val, max_val = datatype.min(), datatype.max()

        # Clip to range and round to integer
        result = np.clip(values, min_val, max_val)
        result = np.round(result)

        return result

    # Note: cppsim and rtlsim execution are handled by HLSBackend.execute_node()
    # We delegate to parent class via super().execute_node() in execute_node() above.

    def global_includes(self):
        """Generate global includes.

        Note: params.hpp is NOT included here because it depends on type definitions
        (LhsType, RhsType, OutType) that are generated in defines(). Instead,
        params.hpp is included in defines() AFTER those type definitions.
        """
        self.code_gen_dict["$GLOBALS$"] = ['#include "flatten.hpp"']

    def defines(self, var):
        """Generate type definitions and constants.

        CRITICAL: Type definitions must come BEFORE params.hpp include because
        params.hpp uses these types (LhsType, RhsType) in array declarations.
        """
        lhs_iface = self.design_point.inputs["lhs"]
        tensor_shape = lhs_iface.tensor_shape
        num_channels = tensor_shape[-1]  # Last dimension is channels
        numReps = tensor_shape[0]  # First dimension is batch
        pe = lhs_iface.stream_shape[-1]  # PE from stream tiling

        # Calculate spatial dimension from tensor shape
        if len(tensor_shape) == 4:  # [N, H, W, C] - image data (NHWC format)
            spatial_dim = tensor_shape[1] * tensor_shape[2]
        elif len(tensor_shape) == 3:  # [N, Seq, C] - sequence data (NLC format)
            spatial_dim = tensor_shape[1]
        elif len(tensor_shape) == 2:  # [N, C] - fully connected
            spatial_dim = 1
        else:
            raise Exception(
                f"Unexpected tensor shape {tensor_shape}. Expected 2D [N,C], 3D [N,Seq,C], or 4D [N,H,W,C]"
            )

        # Get HLS type strings for inputs/outputs
        lhs_hls_type = self.get_input_datatype(0).get_hls_datatype_str()
        rhs_hls_type = self.get_input_datatype(1).get_hls_datatype_str()
        out_hls_type = self.get_output_datatype().get_hls_datatype_str()

        self.code_gen_dict["$DEFINES$"] = [
            # Type definitions MUST come first (params.hpp depends on these)
            f"using LhsType = {lhs_hls_type};",
            f"using RhsType = {rhs_hls_type};",
            f"using OutType = {out_hls_type};",
            # Constant definitions
            # TODO(post-release): Remove unused macros (NumChannels, SpatialDim, numReps)
            # These are computed but never referenced in generated HLS code.
            # Only PE is actually used (in out[PE] declarations and loop bounds).
            # Code generation in docompute() uses dynamic loops from output_shape.
            f"#define NumChannels {num_channels}",  # UNUSED - remove post-release
            f"#define PE {pe}",  # Used in generated code
            f"#define SpatialDim {spatial_dim}",  # UNUSED - remove post-release
            f"#define numReps {numReps}",  # UNUSED - remove post-release
            # Include params.hpp AFTER type definitions
            # (params.hpp contains arrays like: LhsType lhs[...], RhsType rhs[...])
            '#include "params.hpp"',
        ]

    def read_npy_data(self):
        """Read streaming inputs from numpy files for C++ simulation."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$READNPYDATA$"] = []

        # Read LHS if streaming
        if self._needs_streaming_interface("lhs"):
            lhs_dtype = self.get_input_datatype(0)
            lhs_elem_bits = lhs_dtype.bitwidth()
            lhs_packed_bits = self.get_instream_width(ind=0)
            lhs_packed_hls_type = f"ap_uint<{lhs_packed_bits}>"
            lhs_elem_hls_type = lhs_dtype.get_hls_datatype_str()
            npy_type = "float"
            npy_in_lhs = f"{code_gen_dir}/input_0.npy"

            self.code_gen_dict["$READNPYDATA$"].append(
                f"npy2apintstream<{lhs_packed_hls_type}, {lhs_elem_hls_type}, {lhs_elem_bits}, "
                f'{npy_type}>("{npy_in_lhs}", in0_V, false);'
            )

        # Read RHS if streaming (dynamic_dynamic pattern)
        if self._needs_streaming_interface("rhs"):
            rhs_dtype = self.get_input_datatype(1)
            rhs_elem_bits = rhs_dtype.bitwidth()
            rhs_packed_bits = self.get_instream_width(ind=1)
            rhs_packed_hls_type = f"ap_uint<{rhs_packed_bits}>"
            rhs_elem_hls_type = rhs_dtype.get_hls_datatype_str()
            npy_type = "float"
            npy_in_rhs = f"{code_gen_dir}/input_1.npy"

            self.code_gen_dict["$READNPYDATA$"].append(
                f"npy2apintstream<{rhs_packed_hls_type}, {rhs_elem_hls_type}, {rhs_elem_bits}, "
                f'{npy_type}>("{npy_in_rhs}", in1_V, false);'
            )

    def strm_decl(self):
        """Declare HLS streams for C++ simulation based on input pattern."""
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []

        # LHS stream (if streaming)
        if self._needs_streaming_interface("lhs"):
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{self.get_instream_width(ind=0)}>> in0_V ("in0_V");'
            )

        # RHS stream (if streaming in dynamic_dynamic pattern)
        if self._needs_streaming_interface("rhs"):
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{self.get_instream_width(ind=1)}>> in1_V ("in1_V");'
            )

        # Output stream (always present)
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");'
        )

    def docompute(self):
        """Generate main computation with broadcasting support.

        Orchestrates code generation for:
        1. Header and output buffer declaration
        2. Input buffer declarations for broadcasting
        3. Nested loops with stream I/O and PE-parallel operations

        Generated code is assigned to self.code_gen_dict["$DOCOMPUTE$"].
        """
        # Get template parameters and configuration
        tmpl_args = self.get_template_param_values()
        op_str = self.cpp_op
        lhs_is_streaming = self._needs_streaming_interface("lhs")
        rhs_is_streaming = self._needs_streaming_interface("rhs")

        # Get shape information
        lhs_iface = self.design_point.inputs["lhs"]
        pe = lhs_iface.stream_shape[-1]
        output_shape = self.design_point.outputs["output"].tensor_shape
        ndim = len(output_shape) - 1  # Exclude PE dimension
        loop_counters = tuple(f"i{d}" for d in range(ndim))

        # Generate each section
        code = []
        code.extend(self._generate_header(tmpl_args))
        code.extend(self._generate_buffer_declarations(pe, lhs_is_streaming, rhs_is_streaming))
        code.extend(self._generate_loop_headers(output_shape, ndim))
        code.extend(
            self._generate_lhs_stream_read(tmpl_args, loop_counters, ndim, lhs_is_streaming)
        )
        code.extend(
            self._generate_rhs_stream_read(tmpl_args, loop_counters, ndim, rhs_is_streaming)
        )
        code.extend(self._generate_pe_operations(op_str, loop_counters, ndim))
        code.extend(self._generate_loop_closings(ndim))

        # Assign to template placeholder
        self.code_gen_dict["$DOCOMPUTE$"] = code

    def _generate_header(self, tmpl_args: dict) -> list[str]:
        """Generate computation header and output buffer declaration.

        Args:
            tmpl_args: Template parameter values from get_template_param_values()

        Returns:
            List of C++ code lines for header section
        """
        func = self.get_nodeattr("func")

        # Determine pattern for documentation
        lhs_streaming = self._needs_streaming_interface("lhs")
        rhs_streaming = self._needs_streaming_interface("rhs")
        pattern_desc = f"lhs={'stream' if lhs_streaming else 'static'}, rhs={'stream' if rhs_streaming else 'static'}"

        return [
            f"// Elementwise binary operation: {func} ({pattern_desc})",
            f"{tmpl_args['OutType']} out[PE];",
            "#pragma HLS ARRAY_PARTITION variable=out complete dim=1",
            "",
        ]

    def _generate_buffer_declarations(
        self, pe: int, lhs_is_streaming: bool, rhs_is_streaming: bool
    ) -> list[str]:
        """Generate input buffer declarations for broadcasting.

        Declares LHS and RHS buffers with appropriate shapes based on
        broadcast pattern. Static inputs use empty declarations.

        Args:
            pe: Parallelism factor
            lhs_is_streaming: Whether LHS input is streaming
            rhs_is_streaming: Whether RHS input is streaming

        Returns:
            List of C++ code lines for buffer declarations
        """
        code = []

        # LHS buffer declaration (if streaming with broadcasting)
        if lhs_is_streaming:
            lhs_buffer = self._get_buffer_declaration("lhs", pe)
            if lhs_buffer:  # Only if broadcasting requires buffer
                code.append(lhs_buffer.declaration)
                code.append(lhs_buffer.partition_pragma)

        # RHS buffer declaration (if streaming with broadcasting)
        if rhs_is_streaming:
            rhs_buffer = self._get_buffer_declaration("rhs", pe)
            if rhs_buffer:  # Only if broadcasting requires buffer
                code.append(rhs_buffer.declaration)
                code.append(rhs_buffer.partition_pragma)

        if code:
            code.append("")

        return code

    def _generate_loop_headers(self, output_shape: tuple, ndim: int) -> list[str]:
        """Generate nested loop opening statements and pipeline pragma.

        Args:
            output_shape: Output tensor shape (excluding PE dimension)
            ndim: Number of dimensions to loop over

        Returns:
            List of C++ for loop headers with pipeline pragma
        """
        code = []

        # Generate loop nest over output shape dimensions
        for d, size in enumerate(output_shape[:-1]):  # Exclude last (PE) dimension
            indent = "    " * d
            code.append(f"{indent}for(unsigned int i{d} = 0; i{d} < {size}; i{d}++) {{")

        # Pipeline pragma at innermost loop level
        indent = "    " * ndim
        code.append(f"{indent}#pragma HLS pipeline II=1 style=flp")

        return code

    def _generate_lhs_stream_read(
        self, tmpl_args: dict, loop_counters: tuple, ndim: int, lhs_is_streaming: bool
    ) -> list[str]:
        """Generate LHS stream read with unpacking.

        Args:
            tmpl_args: Template parameter values
            loop_counters: Loop variable names
            ndim: Number of dimensions
            lhs_is_streaming: Whether LHS is streaming

        Returns:
            List of C++ code lines for LHS read
        """
        if not lhs_is_streaming:
            return []

        code = []
        indent = "    " * ndim

        lhs_read_cond = self._get_read_condition("lhs", loop_counters)
        lhs_index = self._get_indexing_expression("lhs", loop_counters, "pe")

        code.append(f"{indent}// Read LHS from stream")
        if lhs_read_cond != "true":
            code.append(f"{indent}if({lhs_read_cond}) {{")
            inner_indent = indent + "    "
        else:
            inner_indent = indent

        code.append(f"{inner_indent}const auto lhs_packed = in0_V.read();")
        code.append(f"{inner_indent}const auto lhs_slice = {tmpl_args['LhsSlice']}(lhs_packed);")
        code.append(f"{inner_indent}for(unsigned int pe = 0; pe < PE; pe++) {{")
        code.append(f"{inner_indent}#pragma HLS unroll")
        code.append(f"{inner_indent}    lhs{lhs_index} = lhs_slice(pe, 0);")
        code.append(f"{inner_indent}}}")

        if lhs_read_cond != "true":
            code.append(f"{indent}}}")

        return code

    def _generate_rhs_stream_read(
        self, tmpl_args: dict, loop_counters: tuple, ndim: int, rhs_is_streaming: bool
    ) -> list[str]:
        """Generate RHS stream read with broadcast-aware conditional.

        Args:
            tmpl_args: Template parameter values
            loop_counters: Loop variable names
            ndim: Number of dimensions
            rhs_is_streaming: Whether RHS is streaming

        Returns:
            List of C++ code lines for RHS read
        """
        if not rhs_is_streaming:
            return []

        code = []
        indent = "    " * ndim

        rhs_read_cond = self._get_read_condition("rhs", loop_counters)
        rhs_index = self._get_indexing_expression("rhs", loop_counters, "pe")

        code.append(f"{indent}// Read RHS from stream")
        if rhs_read_cond != "true":
            code.append(f"{indent}if({rhs_read_cond}) {{")
            inner_indent = indent + "    "
        else:
            inner_indent = indent

        code.append(f"{inner_indent}const auto rhs_packed = in1_V.read();")
        code.append(f"{inner_indent}const auto rhs_slice = {tmpl_args['RhsSlice']}(rhs_packed);")
        code.append(f"{inner_indent}for(unsigned int pe = 0; pe < PE; pe++) {{")
        code.append(f"{inner_indent}#pragma HLS unroll")
        code.append(f"{inner_indent}    rhs{rhs_index} = rhs_slice(pe, 0);")
        code.append(f"{inner_indent}}}")

        if rhs_read_cond != "true":
            code.append(f"{indent}}}")

        return code

    def _generate_pe_operations(self, op_str: str, loop_counters: tuple, ndim: int) -> list[str]:
        """Generate PE-parallel computation and output write.

        Args:
            op_str: C++ operation template string
            loop_counters: Loop variable names
            ndim: Number of dimensions

        Returns:
            List of C++ code lines for PE operations
        """
        code = []
        indent = "    " * ndim

        # PE-parallel operations
        code.append(f"{indent}// PE-parallel operations")
        code.append(f"{indent}for(unsigned int pe = 0; pe < PE; pe++) {{")
        code.append(f"{indent}#pragma HLS unroll")

        # Get values from buffers (streaming) or params.hpp (static)
        lhs_index_expr = self._get_indexing_expression("lhs", loop_counters, "pe")
        code.append(f"{indent}    const auto lhs_val = lhs{lhs_index_expr};")

        rhs_index_expr = self._get_indexing_expression("rhs", loop_counters, "pe")
        code.append(f"{indent}    const auto rhs_val = rhs{rhs_index_expr};")

        code.append(f"{indent}    out[pe] = {op_str.format('lhs_val', 'rhs_val')};")
        code.append(f"{indent}}}")

        # Write output
        code.append(f"{indent}// Write output")
        code.append(f"{indent}out0_V.write(flatten(out));")

        return code

    def _generate_loop_closings(self, ndim: int) -> list[str]:
        """Generate nested loop closing braces.

        Args:
            ndim: Number of dimensions

        Returns:
            List of closing brace lines
        """
        code = []
        for d in range(ndim - 1, -1, -1):
            indent = "    " * d
            code.append(f"{indent}}}")
        return code

    def dataoutstrm(self):
        """Write output stream to numpy file for C++ simulation."""
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

        # Folded shape for HLS I/O
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f"apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, "
            f'{npy_type}>(out0_V, {shape_cpp_str}, "{npy_out}", false);'
        ]

    def blackboxfunction(self):
        """Generate function signature for IP generation based on input pattern."""
        # Build parameter list based on which inputs are streaming
        params = []

        if self._needs_streaming_interface("lhs"):
            params.append(f"hls::stream<ap_uint<{self.get_instream_width(ind=0)}>> &in0_V")

        if self._needs_streaming_interface("rhs"):
            params.append(f"hls::stream<ap_uint<{self.get_instream_width(ind=1)}>> &in1_V")

        # Output stream always present
        params.append(f"hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V")

        # Generate function signature
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}({', '.join(params)})"
        ]

    def pragmas(self):
        """Generate HLS interface pragmas based on input pattern."""
        self.code_gen_dict["$PRAGMAS$"] = []

        # Add pragma for each streaming input
        if self._needs_streaming_interface("lhs"):
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in0_V")

        if self._needs_streaming_interface("rhs"):
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")

        # Output stream always present
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")
