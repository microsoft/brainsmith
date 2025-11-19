# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import numpy as np
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from scipy.special import softmax

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM, KernelOp
from brainsmith.dataflow.constraints import IsDynamic
from brainsmith.dataflow.spec_helpers import constant_datatype, derive_dim
from brainsmith.dataflow.types import ShapeHierarchy
from brainsmith.registry import kernel

# Module-level unified KernelSchema (structure + transformation)
SOFTMAX_SCHEMA = df.KernelSchema(
    name="Softmax",
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM],  # One Softmax op: (1, 1, channels)
            stream_tiling=["SIMD"],  # Stream channels with SIMD parallelism
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],  # Same as input: (1, 1, channels)
            stream_tiling=[
                derive_dim("input", ShapeHierarchy.STREAM, -1)
            ],  # Output streams at same rate as input
            datatype=constant_datatype("FLOAT32"),  # Always FLOAT32 (integer inputs upcast in HLS)
        )
    ],
    constraints=[
        # Input must be dynamic (no initializers)
        # Note: Integer inputs (e.g., INT4, INT8) are safely upcast to FLOAT32 in HLS
        IsDynamic("input"),
    ],
)


@kernel(description="Float32 Softmax using Dataflow Modeling", author="Shane Fleming")
class Softmax(KernelOp):
    """Abstraction layer for HW implementation of Softmax layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> df.KernelSchema:
        return SOFTMAX_SCHEMA

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if ONNX node can be converted to Softmax kernel.

        Only accepts Softmax nodes operating on last axis (channel dimension).
        """
        if node.op_type != "Softmax" or node.domain == "brainsmith.kernels":
            return False

        # Check axis attribute (must be None or -1 for channel-wise softmax)
        axis_attr = get_by_name(node.attribute, "axis")
        if axis_attr is None:
            axis = -1  # Default value for Softmax
        else:
            axis = axis_attr.i

        return axis == -1

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int, kernel_index: int = None
    ) -> df.TransformationResult:
        """Create Softmax Kernel node from ONNX Softmax node.

        NOTE: Softmax operates on the last dimension (axis=-1) and is layout-agnostic.
        However, the global normalize_dataflow_layouts preprocessing pass ensures
        inputs are in NHWC layout for consistency with other dataflow kernels.

        Args:
            node: ONNX Softmax node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes
            kernel_index: Sequential index for this kernel type (for naming)
        """
        cls.build_schema(node, model)

        # Create HW node with sequential naming
        node_name = f"Softmax_{kernel_index}" if kernel_index is not None else f"Softmax_{node.name}"
        hw_node = helper.make_node(
            "Softmax",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name=node_name,
        )

        return df.TransformationResult(nodes_to_insert=[hw_node], nodes_to_remove=[node])

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        output_data = softmax(input_data, axis=-1)
        context[node.output[0]] = output_data.astype(np.float32)
