# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import numpy as np
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM, KernelOp
from brainsmith.dataflow.constraints import AttrCompare
from brainsmith.dataflow.spec_helpers import constant_datatype, derive_dim
from brainsmith.dataflow.types import ShapeHierarchy
from brainsmith.registry import kernel

# =============================================================================
# Clean Product Schema
# =============================================================================

LAYERNORM_SCHEMA = df.KernelSchema(
    name="LayerNorm",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],  # (1, 1, channels) - process full spatial dims
            stream_tiling=["SIMD"],  # Stream channels with SIMD parallelism
            required_layout="NHWC",  # Hardware requires NHWC layout
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],  # (1, 1, channels)
            stream_tiling=[
                derive_dim("input", ShapeHierarchy.STREAM, -1)
            ],  # Output streams at same rate as input
            datatype=constant_datatype("FLOAT32"),  # Output datatype same as input
            required_layout="NHWC",  # Hardware produces NHWC layout
        )
    ],
    kernel_params={
        "epsilon": ("f", True, 1e-5),
    },
    constraints=[
        # Product constraint: epsilon must be positive for numerical stability
        AttrCompare("epsilon", ">", 0),
    ],
)


@kernel(description="Hardware LayerNorm w/out Bias/Scale", author="Shane Fleming")
class LayerNorm(KernelOp):
    """Abstraction layer for HW implementation of the LayerNorm layer."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> df.KernelSchema:
        """Build LayerNorm schema (constant for all instances)."""
        return LAYERNORM_SCHEMA

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if ONNX node can be converted to LayerNorm kernel.

        Only accepts FuncLayerNorm nodes operating on last axis (channel dimension).
        """
        if node.op_type != "FuncLayerNorm":
            return False

        # Check axis attribute (must be -1 or None for channel-wise normalization)
        axis_attr = get_by_name(node.attribute, "axis")
        return axis_attr is None or axis_attr.i == -1

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int, kernel_index: int = None
    ) -> df.TransformationResult:
        """Create LayerNorm HW node from FuncLayerNorm node.

        Args:
            node: FuncLayerNorm node
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes (unused - no layout conversion)
            kernel_index: Sequential index for this kernel type (for naming)

        Returns:
            TransformationResult with LayerNorm node
        """
        cls.build_schema(node, model)

        # Extract epsilon from FuncLayerNorm
        epsilon_attr = get_by_name(node.attribute, "epsilon")
        # Pass along None case, handled by kernel schema default
        epsilon = epsilon_attr if epsilon_attr is None else epsilon_attr.f

        # Create HW node with sequential naming
        node_name = f"LayerNorm_{kernel_index}" if kernel_index is not None else f"LayerNorm_{node.name}"
        hw_node = helper.make_node(
            "LayerNorm",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name=node_name,
            epsilon=epsilon,
        )

        return df.TransformationResult(nodes_to_insert=[hw_node], nodes_to_remove=[node])

    def execute_node(self, context, graph):
        node = self.onnx_node
        in_values = context[node.input[0]]

        # Get epsilon from nodeattr
        epsilon = self.get_nodeattr("epsilon")

        # LayerNorm over last dimension (channels)
        # Calculate mean and variance along channel axis
        mean = np.mean(in_values, axis=-1, keepdims=True)
        var = np.var(in_values, axis=-1, keepdims=True)

        # Normalize: (x - mean) / sqrt(var + epsilon)
        normalized = (in_values - mean) / np.sqrt(var + epsilon)

        # Store result
        context[node.output[0]] = normalized.astype(np.float32)
