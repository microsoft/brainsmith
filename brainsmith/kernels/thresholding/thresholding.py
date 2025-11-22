############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
# Refactored to eliminate redundancy and leverage dataflow system (2025)
############################################################################
# ARETE REFACTORING NOTES:
# - Deleted 116+ lines of redundant code
# - Removed shape/stream methods that duplicate KernelOp base class
# - Standardized nodeattr naming (input0Datatype, input1Datatype, output0Datatype)
# - Simplified infer_from() - direct onnx.helper usage, no abstraction layers
# - Kept only HW-specific logic (TMEM calc, threshold tensor formatting, decoupled mode)
# - Trusts the dataflow system instead of manual reimplementation
############################################################################


import logging

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM, KernelOp
from brainsmith.dataflow.constraints import (
    IsDynamic,
)
from brainsmith.dataflow.spec_helpers import derive_dim
from brainsmith.dataflow.types import ShapeHierarchy
from brainsmith.registry import kernel

logger = logging.getLogger(__name__)

# =============================================================================
# Thresholding Schema
# =============================================================================


THRESHOLDING_SCHEMA = df.KernelSchema(
    name="Thresholding",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],  # Process full spatial dimensions
            stream_tiling=["PE"],  # Parallel channels with PE
            required_layout="NHWC",  # Hardware requires NHWC layout
        ),
        df.InputSchema(
            name="thresholds",
            # Thresholds are constant weights (NumChannels x num_steps)
            # Not tiled or streamed - full tensor loaded as initializer
            block_tiling=[],  # No block tiling (static data)
            stream_tiling=[],  # Not streamed (static data)
            datatype=None,  # Read from graph (ImportQONNXQuantization already set it)
            mem_modes=frozenset({"embedded", "decoupled", "dynamic"}),  # All possible modes
        ),
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],  # Same as input
            stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)],  # Match input PE
            datatype=None,  # Datatype comes from ONNX graph (set via node attrs)
            required_layout="NHWC",
        )
    ],
    # =========================================================================
    # KERNEL PARAMETERS: Threshold-specific configuration
    # =========================================================================
    kernel_params={
        "num_steps": ("i", True, 1),  # Number of threshold steps (required)
        "act_val": ("i", False, 0),  # Activation bias value (ActVal)
        "num_input_vectors": ("ints", False, [1]),  # Batch/spatial dims (legacy)
        # REMOVED: runtime_writeable_weights - AXI-lite support removed for simplicity
    },
    # =========================================================================
    # VALIDATION: Constraints
    # =========================================================================
    constraints=[
        IsDynamic(("input",)),
        # Note: IsStatic(("thresholds",)) removed - causes issues in loop bodies
        # where thresholds are streamed. mem_modes handles embedded/decoupled/dynamic.
    ],
    # Parallelization
)


# =============================================================================
# Thresholding Kernel Implementation
# =============================================================================


@kernel(
    description="Hardware multi-threshold activation (KernelOp-based)",
    author="Microsoft Corporation",
)
class Thresholding(KernelOp):  # → HWCustomOp → CustomOp (inheritance chain)
    """Modern Thresholding implementation using KernelOp system.

    This kernel applies multi-threshold activation functions to input tensors.
    It compares each input value against a set of thresholds to produce
    quantized outputs.

    Key features:
    - Schema-driven design (no shape storage)
    - Supports both internal_embedded and internal_decoupled memory modes
    - Parallelization via PE parameter
    - Optional runtime-writable thresholds (internal_decoupled mode)

    Arete principles:
    - Shapes extracted from design_point (not nodeattrs)
    - Declarative constraints in schema
    - Two-phase construction (DesignSpace → Configuration)
    """

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> df.KernelSchema:
        """Build Thresholding schema (constant for all instances)."""
        return THRESHOLDING_SCHEMA

    # ================================================================
    # Inference (Static methods)
    # ================================================================

    @staticmethod
    def can_infer_from(node, model: ModelWrapper) -> bool:
        """Check if MultiThreshold node can convert to Thresholding.

        Only checks source-node attributes that won't be preserved in target node.
        Datatype validation is handled by schema constraint: DatatypeInteger((("input", "output"))).
        """
        if node.op_type != "MultiThreshold":
            return False

        from qonnx.custom_op.registry import getCustomOp

        mt_inst = getCustomOp(node)

        # Check MultiThreshold-specific constraints (not preserved in Thresholding node)
        return mt_inst.get_nodeattr("out_scale") == 1.0 and int(
            mt_inst.get_nodeattr("out_bias")
        ) == mt_inst.get_nodeattr("out_bias")

    @staticmethod
    def infer_from(node, model: ModelWrapper, insert_index: int, kernel_index: int = None) -> df.TransformationResult:
        """Convert MultiThreshold node to Thresholding node.

        Extracts and validates MultiThreshold-specific parameters (scale, actval).

        NOTE: Assumes input is already in NHWC layout (preprocessing required).

        Args:
            node: MultiThreshold ONNX node
            model: Model wrapper
            insert_index: Where to insert new node (unused - no layout conversion)
            kernel_index: Sequential index for this kernel type (for naming)

        Returns:
            df.TransformationResult with new Thresholding node
        """
        from qonnx.custom_op.registry import getCustomOp

        # Extract and validate MultiThreshold parameters
        mt_inst = getCustomOp(node)
        scale = mt_inst.get_nodeattr("out_scale")
        actval = mt_inst.get_nodeattr("out_bias")

        if scale != 1.0:
            raise ValueError(
                f"{node.name}: MultiThreshold out_scale must be 1.0 for HW conversion, got {scale}"
            )

        if int(actval) != actval:
            raise ValueError(
                f"{node.name}: MultiThreshold out_bias must be integer for HW conversion, got {actval}"
            )
        actval = int(actval)

        # Validate actval sign for signed outputs
        odt = model.get_tensor_datatype(node.output[0])
        if odt != DataType["BIPOLAR"] and odt.signed() and actval >= 0:
            raise ValueError(f"{node.name}: Signed output requires actval < 0, got {actval}")

        # Get shapes
        thl_thres_shape = model.get_tensor_shape(node.input[1])
        thl_in_shape = model.get_tensor_shape(node.input[0])

        # Create HW node with sequential naming
        node_name = f"Thresholding_{kernel_index}" if kernel_index is not None else f"Thresholding_{node.name}"
        hw_node = helper.make_node(
            "Thresholding",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name=node_name,
            # Kernel parameters
            num_steps=int(thl_thres_shape[1]),
            act_val=actval,
            num_input_vectors=list(thl_in_shape[:-1]),
            runtime_writeable_weights=0,
        )

        # Mark thresholds as weight (for mem_mode parameter creation)
        # Thresholds input (index 1) is always an initializer
        # Attribute presence indicates weight; builder will create parameter
        thresholds_input = node.input[1]
        if model.get_initializer(thresholds_input) is not None:
            hw_node.attribute.append(helper.make_attribute("input1MemType", "embedded"))

        return df.TransformationResult(nodes_to_insert=[hw_node], nodes_to_remove=[node])

    # ================================================================
    # Custom Stream Width (Decoupled Threshold Memory Mode)
    # ================================================================

    def get_instream_width(self, ind=0):
        """Get input stream width in bits.

        Overrides base class for ind=1 to handle threshold memory modes.
        In decoupled and dynamic modes, thresholds stream in via AXI-Stream.

        For ind=0 (data): Uses base class (PE * input_datatype.bitwidth())
        For ind=1 (thresholds): PE * weight_datatype.bitwidth() * num_steps if streaming, else 0
        """
        if ind == 0:
            # Use base class implementation
            return super().get_instream_width(ind)
        elif ind == 1:
            # Get mem_mode from design point inputs (defaults to "embedded")
            thresholds_iface = self.design_point.inputs.get("thresholds")
            mem_mode = (thresholds_iface.mem_mode if thresholds_iface and thresholds_iface.mem_mode
                       else "embedded")

            # Both decoupled and dynamic modes require streaming interface
            if mem_mode in ("decoupled", "dynamic"):
                pe = self.get_nodeattr("PE")
                wp = self.get_input_datatype(1).bitwidth()
                n_thres_steps = self.get_nodeattr("num_steps")
                return pe * wp * n_thres_steps
            return 0  # embedded mode: no streaming interface
        else:
            raise ValueError(f"Invalid input index: {ind}")

    def calc_tmem(self):
        """Calculate TMEM (threshold memory depth).

        Returns: NumChannels // PE
        """
        ki = self.design_point
        num_channels = ki.inputs["input"].block_shape[-1]
        pe = self.get_nodeattr("PE")
        return num_channels // pe

    def get_exp_cycles(self):
        """Return expected cycles for thresholding operation.

        Formula: Channels/PE × batch_size × fmdim × fmdim
        This is the product of all folded output shape dimensions except the last (PE).

        Returns:
            int: Expected number of cycles
        """
        import numpy as np
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert threshold matrix to HW-compatible format.

        Ensures:
        - NumChannels % PE == 0
        - For unsigned inputs, thresholds are positive
        - Rows interleaved between PEs
        - Reshaped to (PE, TMEM, n_thres_steps)

        Args:
            orig_thres_matrix: Original threshold matrix (NumChannels, n_thres_steps)

        Returns:
            Reshaped threshold tensor (1, PE, TMEM, n_thres_steps)
        """
        ki = self.design_point
        num_channels = ki.inputs["input"].block_shape[-1]
        pe = self.get_nodeattr("PE")
        tmem = num_channels // pe

        assert (
            num_channels % pe == 0
        ), f"Requirement NumChannels={num_channels} divisible by PE={pe} is violated."

        assert orig_thres_matrix.ndim == 2, "Threshold matrix dimension is not as expected (2)."

        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.get_nodeattr("num_steps"), "Mismatch in threshold steps"

        # For unsigned inputs, ensure all thresholds are nonnegative
        if not self.get_input_datatype(0).signed():
            assert (orig_thres_matrix >= 0).all(), "Unsigned input requires nonnegative thresholds"

        ret = orig_thres_matrix

        # Ensure channels match NumChannels, duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (num_channels, 1))

        assert (
            ret.shape[0] == num_channels
        ), f"Channels of threshold matrix ({ret.shape[0]}) don't match NumChannels ({num_channels})"

        # Distribute rows between PEs (interleaving)
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)

        assert (
            ret.shape[0] == pe
        ), f"First dimension after PE distribution ({ret.shape[0]}) != PE ({pe})"
        assert (
            ret.shape[1] == tmem
        ), f"Second dimension after PE distribution ({ret.shape[1]}) != TMEM ({tmem})"
        assert (
            ret.shape[2] == n_thres_steps
        ), f"Third dimension after PE distribution ({ret.shape[2]}) != numSteps ({n_thres_steps})"

        return ret.reshape(1, pe, tmem, n_thres_steps)

    def execute_node(self, context, graph):
        """Execute thresholding operation using QONNX multithreshold.

        Applies multi-threshold activation to input tensor.
        """
        # Ensure design_space initialized (QONNX executor creates fresh instances)
        self._ensure_initialized_for_execution(graph)

        node = self.onnx_node
        inp_values = context[node.input[0]]
        th_val = context[node.input[1]]
        out_bias = self.get_nodeattr("act_val")

        # MultiThreshold expects inputs in (N,C,H,W) or (N,C) format
        # If 4D, input values in context are (N,H,W,C) and need transpose
        # If 2D, inputs can be passed directly
        is_4d = len(inp_values.shape) == 4

        if is_4d:
            inp_values = np.transpose(inp_values, (0, 3, 1, 2))

        # Apply multithreshold
        y = multithreshold(inp_values, th_val, out_bias=out_bias)

        if is_4d:
            y = y.transpose(0, 2, 3, 1)

        # Handle BIPOLAR output (binary to bipolar conversion)
        act = self.get_output_datatype(0)
        if act == DataType["BIPOLAR"]:
            y = 2 * y - 1

        context[node.output[0]] = y.astype(np.float32)

    def infer_node_datatype(self, model):
        """Infer and propagate datatypes (inputs and outputs).

        Overrides base class to also propagate threshold datatype to model.
        Base class only propagates outputs, but threshold dtype optimization
        requires updating the model's input[1] tensor datatype.
        """
        # Call base class (initializes design space, propagates outputs)
        super().infer_node_datatype(model)

        # Additionally propagate threshold datatype to model
        # This matches FINN's minimize_accumulator_width which updates model tensor dtype
        if len(self.onnx_node.input) > 1:
            thresh_dtype = self.get_input_datatype(1)
            model.set_tensor_datatype(self.onnx_node.input[1], thresh_dtype)

    # ================================================================
    # MLO Loop Body Adaptation
    # ================================================================

    def adapt_for_loop_body(self, loop_signature):
        """Adapt Thresholding for use in FINNLoop body.

        Forces threshold memory mode to "dynamic" when weights are streamed from loop level.
        Only modifies the attribute if:
        1. Thresholds are marked as weight (attribute exists from InferKernel)
        2. Loop signature indicates input is PARAMETER (streamed per iteration)

        Args:
            loop_signature: List of LoopBodyInputType values for each input
        """
        from qonnx.util.basic import get_by_name

        # Check if thresholds are marked as weight
        attr = get_by_name(self.onnx_node.attribute, "input1MemType")
        if attr is None:
            return  # Not a weight, nothing to adapt

        # Check if loop signature indicates this input is streamed as parameter
        if loop_signature and len(loop_signature) > 1:
            from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType

            if loop_signature[1] == LoopBodyInputType.PARAMETER:
                self.set_nodeattr("input1MemType", "dynamic")
                logger.debug(f"{self.onnx_node.name}: Forced input1MemType=dynamic for MLO")

    def make_shape_compatible_op(self, model):
        oshape = model.get_tensor_shape(self.onnx_node.output[0])
        # implement tensor with correct shape
        return super().make_const_shape_op(oshape)
