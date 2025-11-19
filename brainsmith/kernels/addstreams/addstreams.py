# Portions derived from FINN project
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AddStreams hardware kernel for element-wise addition of two integer streams
with identical shapes.

Example ONNX pattern:
    Add(input0: INT8[1,224,224,64], input1: INT8[1,224,224,64])
    -> output: INT8[1,224,224,64]

Hardware mapping:
    AddStreams with PE parallelism for channel-wise processing
"""


from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_SHAPE, KernelOp
from brainsmith.dataflow.constraints import DatatypeInteger, IsDynamic, ShapesEqual
from brainsmith.dataflow.spec_helpers import add_datatype, derive_dim
from brainsmith.dataflow.types import ShapeHierarchy
from brainsmith.registry import kernel

ADDSTREAMS_SCHEMA = df.KernelSchema(
    name="AddStreams",
    inputs=[
        df.InputSchema(
            name="input0",
            block_tiling=FULL_SHAPE,  # Rank-agnostic: works with any tensor rank
            stream_tiling=["PE"],
            required_layout="NHWC",  # Embedded layout requirement
        ),
        df.InputSchema(
            name="input1",
            block_tiling=FULL_SHAPE,  # Rank-agnostic: works with any tensor rank
            stream_tiling=["PE"],
            required_layout="NHWC",  # Embedded layout requirement
        ),
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=FULL_SHAPE,  # Rank-agnostic: works with any tensor rank
            stream_tiling=[
                derive_dim("input0", ShapeHierarchy.STREAM, -1)
            ],  # Auto-pads to match rank
            datatype=add_datatype("input0", "input1"),  # INT8 + INT8 → INT9 (prevents overflow)
            required_layout="NHWC",  # Embedded layout requirement
        )
    ],
    constraints=[
        IsDynamic(("input0", "input1")),
        DatatypeInteger(("input0", "input1")),
        ShapesEqual(("input0", "input1")),
    ],
)


@kernel(description="Element-wise addition of two integer streams", author="FINN Team")
class AddStreams(KernelOp):
    """Hardware kernel for element-wise addition of two streams."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> df.KernelSchema:
        """Build AddStreams schema (constant for all instances)."""
        return ADDSTREAMS_SCHEMA

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if ONNX node can be converted to AddStreams kernel."""
        if node.op_type != "Add":
            return False

        # Check we have two inputs
        if len(node.input) != 2:
            return False

        return True

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int, kernel_index: int = None
    ) -> df.TransformationResult:
        """Create AddStreams HW node from ONNX Add node.

        NOTE: Assumes inputs are already in NHWC layout (preprocessing required).

        Args:
            node: ONNX Add node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes (unused - no layout conversion)
            kernel_index: Sequential index for this kernel type (for naming)

        Returns:
            TransformationResult with AddStreams node and removed Add node
        """
        # Create AddStreams HW node with sequential naming
        node_name = f"AddStreams_{kernel_index}" if kernel_index is not None else f"AddStreams_{node.name}"
        hw_node = helper.make_node(
            "AddStreams",
            inputs=list(node.input),
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name=node_name,
        )

        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node],
        )

    def execute_node(self, context, graph):
        """Execute AddStreams on CPU for testing/validation.

        Performs element-wise addition: output = input0 + input1

        Args:
            context: Execution context with tensor values
            graph: ONNX graph
        """
        node = self.onnx_node

        # Get input data
        input0_data = context[node.input[0]]
        input1_data = context[node.input[1]]

        # Element-wise addition
        output_data = input0_data + input1_data

        # Get output datatype and cast
        output_dt = self.get_output_datatype(ind=0)
        output_data = output_data.astype(output_dt.to_numpy_dt())

        # Store result
        context[node.output[0]] = output_data
