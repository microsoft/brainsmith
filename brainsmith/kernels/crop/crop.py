############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Josh Monson <joshmonson@microsoft.com> (original Crop)
# @author       Thomas Keller <thomaskeller@microsoft.com> (AutoCrop migration)
############################################################################

"""Crop hardware kernel using the modern KernelOp system.

This module implements spatial cropping for NHWC tensors using the unified
dataflow system with schema-driven design, declarative constraints, and
two-phase construction for efficient Design Space Exploration.

Key Features:
- **Schema-driven**: All structure defined in CROP_SCHEMA
- **No shape storage**: Extracts shapes from ModelWrapper (never stores)
- **Declarative constraints**: Datatype, divisibility, bounds validation
- **Two-phase construction**: DesignSpace (once) → Configuration (many)
- **Unified transformation**: Can infer from Gather nodes

Example ONNX Pattern:
    Gather(input: INT8[1,224,224,64], indices: [12:212], axis=1)
    -> output: INT8[1,200,224,64]

Hardware Mapping:
    Crop with SIMD parallelism for channel-wise processing
    Crop edges: north/south (height), east/west (width)
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM, KernelOp
from brainsmith.dataflow.constraints import (
    CustomConstraint,
    DatatypeInteger,
    DimensionDivisible,
    DimensionEquals,
    IsDynamic,
)
from brainsmith.dataflow.spec_helpers import derive_dim
from brainsmith.dataflow.types import ShapeHierarchy
from brainsmith.registry import kernel

# =============================================================================
# Helper Functions for Custom Dimension Computation
# =============================================================================


def _compute_output_height(
    interfaces: dict[str, Any], param_getter: Callable, model: Any, tensor_name: str
) -> int:
    """Compute cropped height from input and crop parameters.

    Output height = input height - crop_north - crop_south

    Args:
        interfaces: Dict mapping interface names to InterfaceDesignSpace/Configuration
        param_getter: Function to retrieve nodeattr values
        model: ModelWrapper (unused - for unified signature)
        tensor_name: Tensor name (unused - for unified signature)

    Returns:
        Cropped height dimension (positive integer)

    Raises:
        ValueError: If computed height is invalid
    """
    input_h = interfaces["input"].tensor_shape[1]  # NHWC: [N, H, W, C]
    crop_north = param_getter("crop_north")
    crop_south = param_getter("crop_south")

    output_h = input_h - crop_north - crop_south

    if output_h <= 0:
        raise ValueError(
            f"Invalid cropped height {output_h} "
            f"(input_h={input_h}, crop_north={crop_north}, crop_south={crop_south})"
        )

    return output_h


def _compute_output_width(
    interfaces: dict[str, Any], param_getter: Callable, model: Any, tensor_name: str
) -> int:
    """Compute cropped width from input and crop parameters.

    Output width = input width - crop_east - crop_west

    Args:
        interfaces: Dict mapping interface names to InterfaceDesignSpace/Configuration
        param_getter: Function to retrieve nodeattr values
        model: ModelWrapper (unused - for unified signature)
        tensor_name: Tensor name (unused - for unified signature)

    Returns:
        Cropped width dimension (positive integer)

    Raises:
        ValueError: If computed width is invalid
    """
    input_w = interfaces["input"].tensor_shape[2]  # NHWC: [N, H, W, C]
    crop_east = param_getter("crop_east")
    crop_west = param_getter("crop_west")

    output_w = input_w - crop_east - crop_west

    if output_w <= 0:
        raise ValueError(
            f"Invalid cropped width {output_w} "
            f"(input_w={input_w}, crop_east={crop_east}, crop_west={crop_west})"
        )

    return output_w


def _validate_crop_bounds(ctx) -> str | None:
    """Validate crop bounds are within input dimensions.

    Checks:
    - Crop values are non-negative
    - crop_north + crop_south < input height
    - crop_east + crop_west < input width

    Args:
        ctx: Validation context (DesignSpaceValidationContext or ConfigurationValidationContext)

    Returns:
        Error message if invalid, None if valid
    """
    input_shape = ctx.get_shape("input", df.ShapeHierarchy.TENSOR)
    h, w = input_shape[1], input_shape[2]  # NHWC

    # Get crop parameters (always available in design space / configuration contexts)
    crop_n = ctx.get_param("crop_north")
    crop_s = ctx.get_param("crop_south")
    crop_e = ctx.get_param("crop_east")
    crop_w = ctx.get_param("crop_west")

    # Validate non-negative
    if crop_n < 0 or crop_s < 0 or crop_e < 0 or crop_w < 0:
        return f"Crop values must be non-negative, got (N={crop_n}, S={crop_s}, E={crop_e}, W={crop_w})"

    # Validate height
    if crop_n + crop_s >= h:
        return f"crop_north ({crop_n}) + crop_south ({crop_s}) >= height ({h})"

    # Validate width
    if crop_e + crop_w >= w:
        return f"crop_east ({crop_e}) + crop_west ({crop_w}) >= width ({w})"

    return None  # Valid


# =============================================================================
# Module-Level Schema (Structure + Validation + Transformation)
# =============================================================================

CROP_SCHEMA = df.KernelSchema(
    name="Crop",
    # ========== STRUCTURE ==========
    # Note: Crop hardware kernel has 1 input (data tensor).
    # Gather nodes have 2 inputs (data + indices), but indices are consumed
    # during transformation to compute crop parameters.
    inputs=[
        df.InputSchema(
            name="input",
            block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM, FULL_DIM],  # Full tensor
            stream_tiling=["SIMD"],  # Stream channels with SIMD
            required_layout="NHWC",
        )
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            # Output shape depends on crop parameters - use custom dimension functions
            block_tiling=[
                FULL_DIM,  # N dimension (unchanged)
                _compute_output_height,  # Cropped height
                _compute_output_width,  # Cropped width
                FULL_DIM,  # C dimension (unchanged)
            ],
            stream_tiling=[
                1,
                1,
                1,
                derive_dim("input", ShapeHierarchy.STREAM, -1),
            ],  # Match input SIMD
            datatype="input",  # Pass-through datatype
            required_layout="NHWC",
        )
    ],
    # ========== KERNEL PARAMETERS ==========
    kernel_params={
        "crop_north": ("i", True, 0),  # Rows to remove from top
        "crop_south": ("i", True, 0),  # Rows to remove from bottom
        "crop_east": ("i", True, 0),  # Columns to remove from right
        "crop_west": ("i", True, 0),  # Columns to remove from left
        "channel_fold": ("i", False, 1),  # Batch processing factor
    },
    # ========== VALIDATION ==========
    # Note: These constraints validate the Crop hardware kernel structure (1 input).
    # Additional Gather-specific validation (2 inputs, indices are initializer)
    # is done in can_infer_from() override.
    constraints=[
        # Data input must be dynamic (not an initializer)
        IsDynamic(("input",)),
        # Data input must be integer datatype
        DatatypeInteger(("input",)),
        # Batch size must be 1 (NHWC: dimension 0)
        DimensionEquals("input", 0, 1, hierarchy=df.ShapeHierarchy.TENSOR),
        # SIMD must divide channel dimension (parametric constraint)
        DimensionDivisible("input", -1, "SIMD", hierarchy=df.ShapeHierarchy.STREAM),
        # Custom crop bounds validation
        CustomConstraint(_validate_crop_bounds, "Crop bounds must be within input dimensions"),
    ],
    # ========== TRANSFORMATION ==========
    attribute_mapping={},  # No direct ONNX attrs → kernel params (computed from indices)
)


# =============================================================================
# Crop Kernel Implementation
# =============================================================================


@kernel(
    description="Hardware cropping operation for spatial dimensions (KernelOp version)",
    author="Josh Monson, migrated by Thomas Keller",
)
class Crop(KernelOp):
    """Hardware kernel for 2D spatial cropping with schema-driven design.

    Crops pixels from edges of NHWC tensors:
    - crop_north: Remove rows from top
    - crop_south: Remove rows from bottom
    - crop_east: Remove columns from right
    - crop_west: Remove columns from left

    Schema Auto-Generates:
    - "SIMD" from stream_tiling=[1, 1, 1, "SIMD"]
    - "input0Datatype" from input interface
    - "output0Datatype" from output interface (derived from input)
    - Crop parameters from kernel_params

    Design Space Exploration:
    - SIMD: divisors of input channel dimension
    - channel_fold: batch processing factor

    Example:
        # Create Crop node from ONNX Gather
        node = ... # Gather node with consecutive indices
        model = ... # ModelWrapper
        result = Crop.infer_from(node, model, insert_index=0)

        # DSE: Explore SIMD values
        op = Crop(result.nodes_to_insert[0])
        for config in df.iter_valid_configurations(op, model):
            op.set_nodeattr("SIMD", config["SIMD"])
            ki = op.get_design_point(model)  # Returns KernelDesignPoint
            # Evaluate performance...
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> df.KernelSchema:
        """Build Crop schema (constant for all instances)."""
        return CROP_SCHEMA

    # ================================================================
    # Inference (Custom - needs crop parameter extraction from Gather)
    # ================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if Gather node can be converted to Crop.

        Override default validation to handle Gather-specific requirements:
        - Gather nodes have 2 inputs (data + indices), but Crop has 1
        - Indices must be initializer (static)
        - Indices must be consecutive

        Args:
            node: ONNX Gather node to validate
            model: ModelWrapper for graph context

        Returns:
            True if this Gather can be converted to Crop
        """
        # Check op type
        if node.op_type != "Gather":
            return False

        # Check input count (Gather has 2: data + indices)
        if len(node.input) != 2:
            return False

        # Check output count
        if len(node.output) != 1:
            return False

        # Check that indices input is an initializer (static)
        indices = model.get_initializer(node.input[1])
        if indices is None:
            return False

        # Note: Schema constraints (datatype, dynamic/static, etc.) will be validated
        # during build() after transformation. can_infer_from() only checks ONNX
        # pattern matching (op type, input count, indices are static).

        return True

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int, kernel_index: int = None
    ) -> df.TransformationResult:
        """Create Crop HW node from ONNX Gather node.

        Detects crop patterns from Gather operations:
        - Gather with consecutive indices on spatial axis (height or width)
        - Computes crop_north, crop_south, crop_east, crop_west from indices

        NOTE: Assumes input is already in NHWC layout (preprocessing required).

        Args:
            node: ONNX Gather node to convert
            model: ModelWrapper for graph access
            insert_index: Where to insert new nodes (unused - no layout conversion)
            kernel_index: Sequential index for this kernel type (for naming)

        Returns:
            TransformationResult with Crop node and removed Gather node

        Raises:
            ValueError: If Gather pattern is invalid for crop conversion
        """
        schema = cls.build_schema(node, model)

        # Extract input shape for crop computation (assumes NHWC layout - preprocessing required)
        input_shape = model.get_tensor_shape(node.input[0])

        # Extract indices from Gather (must be initializer)
        indices = model.get_initializer(node.input[1])
        if indices is None:
            raise ValueError(
                f"Gather node '{node.name}' has non-constant indices. "
                "Crop conversion requires static indices."
            )

        # Get axis attribute
        axis_attr = get_by_name(node.attribute, "axis")
        axis = axis_attr.i if axis_attr else 0

        # Normalize negative axis
        if axis < 0:
            axis = len(input_shape) + axis

        # Validate axis is spatial dimension in NHWC
        if axis not in [1, 2]:
            raise ValueError(
                f"Gather axis {axis} not supported for Crop. "
                "Expected axis=1 (height) or axis=2 (width) in NHWC layout."
            )

        # Extract indices and validate consecutive
        indices_flat = indices.flatten()
        min_idx = int(np.min(indices_flat))
        max_idx = int(np.max(indices_flat))

        # Check consecutive (simple approach: length equals range)
        expected_len = max_idx - min_idx + 1
        if len(indices_flat) != expected_len:
            raise ValueError(
                f"Gather indices are not consecutive. "
                f"Expected {expected_len} indices for range [{min_idx}, {max_idx}], "
                f"got {len(indices_flat)}."
            )

        # Compute crop parameters based on axis
        # NHWC: [N=0, H=1, W=2, C=3]
        if axis == 1:  # Height (vertical crop)
            crop_north = min_idx
            crop_south = input_shape[axis] - max_idx - 1
            crop_east = 0
            crop_west = 0
        elif axis == 2:  # Width (horizontal crop)
            crop_north = 0
            crop_south = 0
            crop_east = input_shape[axis] - max_idx - 1
            crop_west = min_idx
        else:
            # Should not reach here due to earlier validation
            raise ValueError(f"Unsupported axis {axis}")

        # Create HW node with crop parameters and sequential naming
        node_name = f"Crop_{kernel_index}" if kernel_index is not None else f"Crop_{node.name}"
        hw_node = helper.make_node(
            "Crop",
            inputs=list(node.input[:1]),  # Only first input (data, not indices)
            outputs=list(node.output),
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name=node_name,
            crop_north=int(crop_north),
            crop_south=int(crop_south),
            crop_east=int(crop_east),
            crop_west=int(crop_west),
            channel_fold=1,
        )

        return df.TransformationResult(
            nodes_to_insert=[hw_node],
            nodes_to_remove=[node],
            metadata={
                "schema_name": schema.name,
                "source_pattern": "Gather",
                "crop_bounds": (crop_north, crop_south, crop_east, crop_west),
                "axis": axis,
            },
        )

    # ================================================================
    # Execution (Reference Implementation)
    # ================================================================

    def execute_node(self, context, graph):
        """Reference numpy execution for testing.

        Applies crop to NHWC input tensor:
        - Crops height dimension: [crop_north : H - crop_south]
        - Crops width dimension: [crop_west : W - crop_east]

        Args:
            context: Execution context (dict mapping tensor names to numpy arrays)
            graph: ONNX graph (not used)
        """
        node = self.onnx_node
        inp = context[node.input[0]]

        # Extract crop parameters from nodeattrs
        crop_n = self.get_nodeattr("crop_north")
        crop_s = self.get_nodeattr("crop_south")
        crop_e = self.get_nodeattr("crop_east")
        crop_w = self.get_nodeattr("crop_west")

        # Apply crop (NHWC layout: [N, H, W, C])
        h, w = inp.shape[1], inp.shape[2]
        h_start, h_end = crop_n, h - crop_s
        w_start, w_end = crop_w, w - crop_e

        # Numpy slicing
        output = inp[:, h_start:h_end, w_start:w_end, :]

        # Store result
        context[node.output[0]] = output

    # ================================================================
    # FINN Compatibility
    # ================================================================

    def make_shape_compatible_op(self, model_w):
        """Create Crop-compatible op for shape inference.

        Returns a Crop node matching legacy behavior for FINN's shape inference.
        This dummy node is used during graph transformations to propagate shapes,
        not for actual execution.

        The base KernelOp class uses auto-detection which returns "RandomNormal",
        but legacy Crop returns "Crop". We override to match legacy behavior for
        parity testing.

        Args:
            model_w: ModelWrapper for ONNX graph access

        Returns:
            ONNX NodeProto with op_type="Crop" for shape inference
        """
        from onnx import helper

        # Extract shapes directly from model (Arete principle - no storage)
        in_shape = model_w.get_tensor_shape(self.onnx_node.input[0])
        out_shape = model_w.get_tensor_shape(self.onnx_node.output[0])

        return helper.make_node(
            "Crop",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            in_shape=list(in_shape),
            out_shape=list(out_shape),
        )
