############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Helper functions for building kernel schemas.

This module provides utilities for creating DimSpec and DatatypeSpec functions
used in InputSchema/OutputSchema definitions. These helpers make it easy to
derive dimensions and datatypes from other interfaces or compute them based on
operation semantics.

Key Components:
- Dimension derivation: derive_dim() for copying dimensions between interfaces
- Datatype derivation: derive_datatype() for copying datatypes
- Value-optimized datatypes: value_optimized_datatype() for static tensor optimization
- Arithmetic range computation: compute_*_range() for binary operations
- Context-aware datatype builders: add_datatype(), mul_datatype(), etc. (auto-detect static vs dynamic)

Design Philosophy (Arete):
- Functions, not classes - pure, composable primitives
- Separate concerns - range computation vs. datatype selection
- Two-layer API: primitives (building blocks) + builders (common patterns)
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qonnx.core.datatype import BaseDataType

# Import from types to access ShapeHierarchy
from .types import ShapeHierarchy

# =============================================================================
# Dimension Derivation Helpers
# =============================================================================


def derive_dim(
    interface: str, hierarchy: ShapeHierarchy, dim_idx: int
) -> Callable[[dict[str, Any], Callable, Any, str | None], int]:
    """Create dimension derivation function (for use in DimSpec).

    Derives dimension value from another interface's shape at specified hierarchy.

    Args:
        interface: Source interface name (e.g., "input", "input0")
        hierarchy: Shape hierarchy level (TENSOR, BLOCK, STREAM)
        dim_idx: Dimension index to copy (supports negative indexing like Python lists)

    Returns:
        Callable with unified signature (interfaces, param_getter, model, tensor_name) -> int

    Example:
        # Explicit function-based derivation
        from brainsmith.dataflow import derive_dim, ShapeHierarchy
        block_tiling=[derive_dim("input", ShapeHierarchy.BLOCK, -1)]

        # Shorthand for common case (STREAM hierarchy, last dimension)
        stream_tiling=[("input", -1)]
    """

    def resolver(
        interfaces: dict[str, Any],
        param_getter: Callable,
        model: Any,  # ModelWrapper - unused but part of unified signature
        tensor_name: str | None,  # Unused but part of unified signature
    ) -> int:
        """Resolve dimension value from interface shape (unified 4-param signature)."""
        if interface not in interfaces:
            available = list(interfaces.keys())
            raise ValueError(
                f"Interface '{interface}' not found for dimension derivation. "
                f"Available: {', '.join(available)}"
            )

        interface_obj = interfaces[interface]

        # Handle both InterfaceDesignPoint (has get_shape) and InterfaceDesignSpace (direct attrs)
        if hasattr(interface_obj, "get_shape"):
            # InterfaceDesignPoint - use get_shape method
            shape = interface_obj.get_shape(hierarchy)
        else:
            # InterfaceDesignSpace - use direct attributes
            if hierarchy == ShapeHierarchy.TENSOR:
                shape = interface_obj.tensor_shape
            elif hierarchy == ShapeHierarchy.BLOCK:
                shape = interface_obj.block_shape
            elif hierarchy == ShapeHierarchy.STREAM:
                raise ValueError(
                    "Cannot derive stream shape from InterfaceDesignSpace. "
                    "Stream shapes require configuration (use get_shape on InterfaceDesignPoint)."
                )
            else:
                raise ValueError(f"Invalid hierarchy: {hierarchy}")

        # Handle negative indexing
        idx = dim_idx if dim_idx >= 0 else len(shape) + dim_idx

        if not (0 <= idx < len(shape)):
            raise ValueError(
                f"Dimension index {dim_idx} out of range for shape {shape} "
                f"(interface '{interface}')"
            )

        return shape[idx]

    return resolver


# =============================================================================
# Datatype Derivation Helpers
# =============================================================================


def derive_datatype(interface: str) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Create datatype derivation function (for use in DatatypeSpec).

    Derives datatype from another interface's datatype.

    Args:
        interface: Source interface name (e.g., "input", "input0")

    Returns:
        Callable that resolves datatype from interfaces dict

    Example:
        # Explicit function-based derivation
        datatype=derive_datatype("input")

        # String shorthand (recommended)
        datatype="input"
    """

    def resolver(
        interfaces: dict[str, Any],
        param_getter: Callable,
        model: Any,  # ModelWrapper
        tensor_name: str,
    ) -> "BaseDataType":
        """Resolve datatype from interface."""
        if interface not in interfaces:
            available = list(interfaces.keys())
            raise ValueError(
                f"Interface '{interface}' not found for datatype derivation. "
                f"Available: {', '.join(available)}"
            )

        return interfaces[interface].datatype

    return resolver


def constant_datatype(datatype_name: str) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Create constant datatype function (for use in DatatypeSpec).

    Returns a fixed DataType regardless of context. Useful for operations that
    always output a specific datatype (e.g., Softmax always outputs FLOAT32).

    Args:
        datatype_name: DataType name (e.g., "FLOAT32", "INT8", "UINT4")

    Returns:
        Callable that returns the constant DataType

    Example:
        # Use constant_datatype for fixed output types
        from brainsmith.dataflow.spec_helpers import constant_datatype
        datatype=constant_datatype("FLOAT32")  # Always outputs FLOAT32

        # Common use case: Softmax always outputs float
        outputs=[
            df.OutputSchema(
                name="output",
                block_tiling=[FULL_DIM],
                stream_tiling=[("input", -1)],
                datatype=constant_datatype("FLOAT32"),  # Input Int4 → Output FLOAT32
            )
        ]
    """
    from qonnx.core.datatype import DataType

    # Validate datatype name at schema definition time
    try:
        dt = DataType[datatype_name]
    except KeyError:
        available = [name for name in dir(DataType) if not name.startswith("_")]
        raise ValueError(
            f"Invalid datatype name: '{datatype_name}'. " f"Available: {', '.join(available)}"
        )

    def resolver(
        interfaces: dict[str, Any],
        param_getter: Callable,
        model: Any,  # ModelWrapper
        tensor_name: str,
    ) -> "BaseDataType":
        """Return constant DataType (context-independent)."""
        return dt

    return resolver


def value_optimized_datatype() -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Create value-optimized datatype function (for use in DatatypeSpec).

    Derives optimal datatype from actual tensor values (static inputs only).
    Falls back to graph datatype if tensor is dynamic (no initializer).

    Returns:
        Callable that resolves datatype from tensor values or graph

    Example:
        # Sentinel constant (recommended)
        from brainsmith.dataflow import VALUE_OPTIMIZED
        datatype=VALUE_OPTIMIZED

        # Explicit function-based
        datatype=value_optimized_datatype()
    """

    def resolver(
        interfaces: dict[str, Any],
        param_getter: Callable,
        model: Any,  # ModelWrapper
        tensor_name: str,
    ) -> "BaseDataType":
        """Resolve optimal datatype from tensor values or fall back to graph."""
        from qonnx.core.datatype import DataType

        if model is None or tensor_name is None:
            raise ValueError(
                "ValueOptimizedDatatype requires model and tensor_name. "
                "This indicates a builder error."
            )

        # Check if static (has initializer)
        initializer = model.get_initializer(tensor_name)
        if initializer is None:
            # Dynamic - use graph datatype
            return model.get_tensor_datatype(tensor_name)

        # Static - optimize from values
        min_val = float(initializer.min())
        max_val = float(initializer.max())

        # Use QONNX utility to find smallest fitting datatype
        if min_val < 0:
            # Signed type - use the more negative extreme value
            extreme = min_val if abs(min_val) > max_val else -max_val - 1
            optimal_dt = DataType.get_smallest_possible(extreme)
        else:
            # Unsigned type sufficient
            optimal_dt = DataType.get_smallest_possible(max_val)

        return optimal_dt

    return resolver


# =============================================================================
# Arithmetic Range Computation (Primitives)
# =============================================================================


def compute_add_range(
    a_min: float, a_max: float, b_min: float, b_max: float
) -> tuple[float, float]:
    """Compute output range for addition: a + b.

    Args:
        a_min, a_max: Bounds for first operand
        b_min, b_max: Bounds for second operand

    Returns:
        (out_min, out_max) tuple

    Example:
        INT8[-128,127] + INT8[-128,127] = [-256, 254] → needs INT9
    """
    return (a_min + b_min, a_max + b_max)


def compute_sub_range(
    a_min: float, a_max: float, b_min: float, b_max: float
) -> tuple[float, float]:
    """Compute output range for subtraction: a - b.

    Args:
        a_min, a_max: Bounds for first operand
        b_min, b_max: Bounds for second operand

    Returns:
        (out_min, out_max) tuple

    Example:
        UINT8[0,255] - [10,20] = [-20,245] → needs signed INT9
    """
    return (a_min - b_max, a_max - b_min)


def compute_mul_range(
    a_min: float, a_max: float, b_min: float, b_max: float
) -> tuple[float, float]:
    """Compute output range for multiplication: a * b.

    Checks all 4 corners due to sign changes.

    Args:
        a_min, a_max: Bounds for first operand
        b_min, b_max: Bounds for second operand

    Returns:
        (out_min, out_max) tuple

    Example:
        INT8[-128,127] * INT4[-8,7] → check all combinations
    """
    corners = [a_min * b_min, a_min * b_max, a_max * b_min, a_max * b_max]
    return (min(corners), max(corners))


def compute_min_range(
    a_min: float, a_max: float, b_min: float, b_max: float
) -> tuple[float, float]:
    """Compute output range for min(a, b).

    Args:
        a_min, a_max: Bounds for first operand
        b_min, b_max: Bounds for second operand

    Returns:
        (out_min, out_max) tuple
    """
    return (min(a_min, b_min), min(a_max, b_max))


def compute_max_range(
    a_min: float, a_max: float, b_min: float, b_max: float
) -> tuple[float, float]:
    """Compute output range for max(a, b).

    Args:
        a_min, a_max: Bounds for first operand
        b_min, b_max: Bounds for second operand

    Returns:
        (out_min, out_max) tuple
    """
    return (max(a_min, b_min), max(a_max, b_max))


def smallest_datatype_for_range(min_val: float, max_val: float) -> "BaseDataType":
    """Find smallest integer datatype that fits the given range.

    Consolidates the logic used throughout the codebase for selecting
    optimal datatypes based on value ranges.

    Uses array-based checking to correctly handle signed/unsigned detection,
    matching FINN's proven approach.

    Args:
        min_val: Minimum value to represent
        max_val: Maximum value to represent

    Returns:
        Smallest QONNX DataType that can represent the range

    Example:
        dt = smallest_datatype_for_range(-100, 127)  # Returns DataType["INT8"]
        dt = smallest_datatype_for_range(0, 255)     # Returns DataType["UINT8"]
        dt = smallest_datatype_for_range(-1016, 1024)  # Returns DataType["INT12"]
    """
    import numpy as np
    from qonnx.core.datatype import DataType

    # Create array with both bounds (matches FINN's approach)
    # This ensures we correctly detect when range includes negative values
    vals = np.array([min_val, max_val], dtype=np.float64)

    # Verify values are integers
    for v in vals:
        assert int(v) == v, f"Non-integer value in range: {v}"

    # Iterate through accumulator candidates (sorted by size, prefers unsigned)
    for k in DataType.get_accumulator_dt_cands():
        dt = DataType[k]

        # Skip unsupported types
        if dt in [DataType["BIPOLAR"], DataType["TERNARY"], DataType["FLOAT32"]]:
            continue

        # Check if datatype can represent BOTH bounds
        if (dt.min() <= vals).all() and (vals <= dt.max()).all():
            return dt

    # Fallback for extreme ranges
    if min_val >= 0:
        return DataType["UINT64"]
    else:
        return DataType["INT64"]


# =============================================================================
# Context-Aware Datatype Builders
# =============================================================================


def _get_bounds(interface_obj, model) -> tuple[float, float]:
    """Extract min/max bounds from interface (auto-detects static vs dynamic).

    Static (has initializer): uses actual measured values
    Dynamic (no initializer): uses datatype bounds (conservative)

    Args:
        interface_obj: Interface design point/space object
        model: ModelWrapper for accessing initializers

    Returns:
        (min_val, max_val) tuple
    """
    # Check if static (has initializer)
    tensor_name = getattr(interface_obj, "tensor_name", None)
    values = model.get_initializer(tensor_name) if tensor_name else None

    if values is not None:
        # Static: use actual measured values
        return float(np.min(values)), float(np.max(values))
    else:
        # Dynamic: use datatype bounds (conservative)
        dt = interface_obj.datatype
        return dt.min(), dt.max()


def _binary_op_datatype(
    a_interface: str,
    b_interface: str,
    range_fn: Callable[[float, float, float, float], tuple[float, float]],
) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Generic binary operation datatype builder.

    Automatically optimizes based on whether interfaces are static or dynamic:
    - Both dynamic: uses worst-case type bounds
    - One static: uses type bounds + actual values
    - Both static: uses actual values for both

    Args:
        a_interface: First operand interface name
        b_interface: Second operand interface name
        range_fn: Function to compute output range (e.g., compute_add_range)

    Returns:
        Callable for OutputSchema datatype field

    Example:
        # AddStreams (both dynamic)
        datatype=add_datatype("input0", "input1")
        # → INT8 + INT8 → INT9 (conservative)

        # ChannelwiseOp (input dynamic, param static)
        datatype=add_datatype("input", "bias")
        # → INT8 + bias[2.0,3.5] → INT8 (optimized)
    """

    def resolver(interfaces, param_getter, model, tensor_name):
        a_min, a_max = _get_bounds(interfaces[a_interface], model)
        b_min, b_max = _get_bounds(interfaces[b_interface], model)

        min_val, max_val = range_fn(a_min, a_max, b_min, b_max)
        return smallest_datatype_for_range(min_val, max_val)

    return resolver


def add_datatype(
    a_interface: str, b_interface: str
) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Compute addition output datatype (context-aware)."""
    return _binary_op_datatype(a_interface, b_interface, compute_add_range)


def sub_datatype(
    a_interface: str, b_interface: str
) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Compute subtraction output datatype (context-aware)."""
    return _binary_op_datatype(a_interface, b_interface, compute_sub_range)


def mul_datatype(
    a_interface: str, b_interface: str
) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Compute multiplication output datatype (context-aware)."""
    return _binary_op_datatype(a_interface, b_interface, compute_mul_range)


def min_datatype(
    a_interface: str, b_interface: str
) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Compute min() output datatype (context-aware)."""
    return _binary_op_datatype(a_interface, b_interface, compute_min_range)


def max_datatype(
    a_interface: str, b_interface: str
) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Compute max() output datatype (context-aware)."""
    return _binary_op_datatype(a_interface, b_interface, compute_max_range)


def threshold_datatype(input_interface: str) -> Callable[[dict, Callable, Any, str], "BaseDataType"]:
    """Compute threshold datatype with ceil() rounding accommodation (FINN-compatible).

    Implements FINN's RoundAndClipThresholds algorithm:
    1. Thresholds undergo ceil() rounding: ceil(127.1) = 128
    2. Clipped to [input.min, input.max + 1]
    3. Therefore threshold dtype must accommodate input.max + 1

    Mathematical rationale:
    - Thresholds divide input domain into quantization bins
    - After ceil(), highest threshold can reach input.max + 1
    - This value represents the supremum (least upper bound) of input domain
    - For INT8 input (-128 to 127), thresholds need INT9 to hold 128

    Args:
        input_interface: Name of input interface (e.g., "input")

    Returns:
        Callable that resolves threshold datatype from input dtype

    Example:
        # In Thresholding schema
        df.InputSchema(
            name="thresholds",
            datatype=threshold_datatype("input"),  # Accommodates ceil() rounding
        )

        # Result: INT8 input → INT9 thresholds (to hold value 128)
    """

    def resolver(
        interfaces: dict[str, Any],
        param_getter: Callable,
        model: Any,  # ModelWrapper
        tensor_name: str,
    ) -> "BaseDataType":
        """Resolve threshold datatype with ceil() accommodation."""
        from qonnx.core.datatype import DataType

        if input_interface not in interfaces:
            available = list(interfaces.keys())
            raise ValueError(
                f"Interface '{input_interface}' not found for threshold datatype. "
                f"Available: {', '.join(available)}"
            )

        input_dt = interfaces[input_interface].datatype

        # Skip optimization for float inputs (keep as-is)
        if not input_dt.is_integer():
            return model.get_tensor_datatype(tensor_name) if model else input_dt

        # Accommodate ceil() rounding: thresholds can reach input.max + 1
        # This is the supremum of the input domain
        max_val = input_dt.max() + 1

        # Find smallest dtype that can hold the ceil-rounded range
        if input_dt.signed():
            # For signed inputs: range is [input.min, input.max+1]
            # Use get_smallest_possible with negative bound for signed dtype selection
            return smallest_datatype_for_range(-max_val - 1, max_val)
        else:
            # For unsigned inputs: range is [0, input.max+1]
            return smallest_datatype_for_range(0, max_val)

    return resolver
