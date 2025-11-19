# Portions derived from FINN project
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ChannelwiseOp hardware kernel using modern KernelOp system."""


import numpy as np
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_SHAPE, KernelOp
from brainsmith.dataflow.constraints import (
    DatatypeInteger,
    IsDynamic,
    IsStatic,
    TensorDimMatches,
    TensorSizeMatches,
)
from brainsmith.dataflow.inference_helpers import (
    expand_scalar_to_channels,
    find_static_dynamic_pair,
)
from brainsmith.dataflow.spec_helpers import (
    add_datatype,
    derive_dim,
    mul_datatype,
    smallest_datatype_for_range,
)
from brainsmith.dataflow.transformation import TransformationResult
from brainsmith.dataflow.types import VALUE_OPTIMIZED, ShapeHierarchy
from brainsmith.registry import kernel


def _channelwise_output_datatype():
    """Polymorphic datatype resolver for ChannelwiseOp based on 'func' nodeattr."""

    def resolver(interfaces, param_getter, model, tensor_name):
        # Handle FINN's uppercase "Func" attribute for compatibility
        try:
            func = param_getter("func")
        except Exception:
            func = param_getter("Func")

        if func == "Add":
            return add_datatype("input", "parameters")(interfaces, param_getter, model, tensor_name)
        elif func == "Mul":
            return mul_datatype("input", "parameters")(interfaces, param_getter, model, tensor_name)
        elif func in ("LessOrEqual", "GreaterOrEqual"):
            return smallest_datatype_for_range(0, 1)
        else:
            raise ValueError(f"Unsupported func '{func}' in ChannelwiseOp")

    return resolver


CHANNELWISE_SCHEMA = df.KernelSchema(
    name="ChannelwiseOp",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=FULL_SHAPE,  # Process full tensor shape
            stream_tiling=["PE"],  # Channel parallelism with PE
            required_layout="NHWC",  # Hardware requires NHWC
        ),
        df.InputSchema(
            name="parameters",
            block_tiling=[],  # No tiling (static data)
            stream_tiling=[],  # Not streamed
            datatype=VALUE_OPTIMIZED,  # Optimize from actual values
            mem_modes=frozenset({"embedded"}),  # Only embedded mode (FINN parity)
        ),
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=FULL_SHAPE,  # Same as input
            stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)],  # Match input PE
            datatype=_channelwise_output_datatype(),  # Dispatches based on func nodeattr
            required_layout="NHWC",
        )
    ],
    # STRUCTURAL (fixed at inference)
    kernel_params={
        # Operation type: ONNX op names
        "func": ("s", True, "Add", {"Add", "Mul", "LessOrEqual", "GreaterOrEqual"}),
    },
    # DSE PARAMETERS (explorable resource parameters)
    dse_parameters={
        # RAM style for parameter storage (HLS-specific)
        "ram_style": df.ParameterSpec(
            name="ram_style", values={"distributed", "block"}, default="distributed"
        ),
    },
    constraints=[
        IsDynamic(("input",)),  # Streaming activation input
        IsStatic(("parameters",)),  # Static weight input
        DatatypeInteger(("input", "parameters")),
        # Parameters: scalar (1) or per-channel (matches input channels)
        TensorSizeMatches("parameters", [1, ("input", -1)]),
        TensorDimMatches("parameters", -1, [1, ("input", -1)]),
    ],
)

# =============================================================================
# ChannelwiseOp Kernel Implementation
# =============================================================================


@kernel(
    description="Channel-wise parametric operations (add/mul/cmp) with PE parallelism",
    author="Thomas Keller (migrated from AMD FINN)",
)
class ChannelwiseOp(KernelOp):
    """Hardware kernel for channel-wise parametric operations.

    Applies channel-wise operations (Add, Mul, LessOrEqual, GreaterOrEqual) to input tensor
    using static parameter tensor.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> df.KernelSchema:
        return CHANNELWISE_SCHEMA

    # ================================================================
    # ONNX → KernelOp Inference (Unified System)
    # ================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        if node.op_type not in ["Add", "Mul", "LessOrEqual", "GreaterOrEqual"]:
            return False

        if len(node.input) != 2:
            return False

        # Check structural pattern: 1 dynamic + 1 static (STRUCTURAL property)
        pair = find_static_dynamic_pair(node.input, model)
        if pair is None:
            return False

        return True

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int, kernel_index: int = None
    ) -> TransformationResult:
        """Infer ChannelwiseOp from ONNX Add/Mul/LessOrEqual/GreaterOrEqual.

        Uses helper functions to detect and reorder inputs to canonical
        (dynamic, static) order before creating HW node.

        Args:
            node: ONNX node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes
            kernel_index: Sequential index for this kernel type (for naming)
        """
        # Detect and reorder inputs to (dynamic, static)
        pair = find_static_dynamic_pair(node.input, model)
        if pair is None:
            raise ValueError(f"Node {node.name} doesn't match ChannelwiseOp pattern")

        dynamic_input, static_input = pair

        # Handle scalar broadcasting if needed
        data_shape = model.get_tensor_shape(dynamic_input)
        param_shape = model.get_tensor_shape(static_input)
        num_channels = data_shape[-1]

        param_input = static_input
        if int(np.prod(param_shape)) == 1:
            # Expand scalar to per-channel
            param_input = expand_scalar_to_channels(static_input, num_channels, model)

        # Create ChannelwiseOp node in canonical order with sequential naming
        node_name = f"ChannelwiseOp_{kernel_index}" if kernel_index is not None else node.name
        hw_node = helper.make_node(
            "ChannelwiseOp",
            inputs=[dynamic_input, param_input],  # Canonical order
            outputs=node.output,
            name=node_name,
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            # Kernel parameters (use ONNX op type directly)
            func=node.op_type,
        )

        # Return transformation result
        # Layout requirements are enforced by schema, not transformation
        return TransformationResult(
            nodes_to_remove=[node],
            nodes_to_insert=[hw_node],
        )

    # ================================================================
    # HW-Specific Methods (Kept - Pattern C Justification)
    # ================================================================

    def calc_tmem(self) -> int:
        """Calculate TMEM (parameter memory depth).

        HW-Specific Logic: Memory depth = total_channels / PE
        Builder guarantees: PE divides tensor_shape[-1]
        """
        input_iface = self.design_point.inputs["input"]
        # TMEM = total channels / PE = (tensor / stream) on channel dimension
        return input_iface.tensor_blocks_shape[-1] * input_iface.stream_cycles_shape[-1]

    def get_hls_compatible_parameter_tensor(self, orig_param_vector):
        """Convert parameter vector to HW-compatible format.

        HW-Specific Logic: Interleaves parameter rows between PEs
        for parallel access. This is not shape computation - it's
        HW memory layout formatting.
        """
        input_iface = self.design_point.inputs["input"]
        num_channels = input_iface.tensor_shape[-1]
        pe = input_iface.stream_shape[-1]
        tmem = self.calc_tmem()

        # Basic shape validation (builder ensures divisibility)
        assert (
            orig_param_vector.ndim == 1
        ), f"Parameter vector dimension is {orig_param_vector.ndim}. Expected dimension: 1."
        assert (
            orig_param_vector.shape[0] == num_channels
        ), f"Parameter vector size {orig_param_vector.shape[0]} != NumChannels {num_channels}"

        # Ensure all parameters are integers (Defensive check)
        assert (
            orig_param_vector.astype(np.int32) == orig_param_vector
        ).all(), "All parameters must be integers"

        # Distribute rows between PEs (interleaving)
        # Transform: [ch0, ch1, ..., ch_n] → [[ch0, ch_PE, ...], [ch1, ch_PE+1, ...], ...]
        ret = orig_param_vector.reshape(tmem, pe).transpose()

        return ret.reshape(1, pe, tmem)

    # ================================================================
    # ONNX Shape Compatibility
    # ================================================================

    def make_shape_compatible_op(self, model):
        """Make ChannelwiseOp compatible with ONNX shape inference."""
        from onnx import helper

        node = self.onnx_node

        # func is already the ONNX op type, use it directly
        return helper.make_node(
            self.get_nodeattr("func"),
            inputs=[node.input[0], node.input[1]],
            outputs=[node.output[0]],
            name=node.name,
        )

    # ================================================================
    # Execution (Reference Implementation)
    # ================================================================

    def execute_node(self, context, graph):
        """Execute channel-wise operation using NumPy."""
        node = self.onnx_node
        func = self.get_nodeattr("func")

        inp_values = context[node.input[0]]
        param_values = context[node.input[1]]

        # Execute operation directly with NumPy
        if func == "Add":
            result = inp_values + param_values
        elif func == "Mul":
            result = inp_values * param_values
        elif func == "LessOrEqual":
            result = inp_values <= param_values
        elif func == "GreaterOrEqual":
            result = inp_values >= param_values
        else:
            raise ValueError(f"Unknown func '{func}'")

        context[node.output[0]] = result.astype(np.float32)

    # ================================================================
    # MLO Loop Body Adaptation
    # ================================================================

    def adapt_for_loop_body(self, loop_signature):
        """Adapt ChannelwiseOp for use in FINNLoop body.

        ChannelwiseOp doesn't require adaptation - parameters remain static
        even in MLO context. Unlike ElementwiseBinaryOp, there's no pattern
        switching needed.

        Args:
            loop_signature: Loop signature describing streaming parameters (unused)
        """
        # No-op for ChannelwiseOp - parameters are always static
        # This method exists for interface compatibility with FINNLoop
        pass
