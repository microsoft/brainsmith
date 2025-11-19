# Portions derived from FINN project
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ElementwiseBinaryOp hardware kernel using modern KernelOp system.

This module implements a polymorphic kernel that handles all elementwise binary
operations (Add, Sub, Mul, Div, logical, comparison, bitwise) with a single
kernel class. Operations are distinguished by the 'func' parameter.

Supported Input Patterns:
- Dynamic + Static (Phase 1): One streaming input, one static parameter
- Dynamic + Dynamic (Phase 2): Both inputs streaming with broadcasting support

Broadcasting Support (Phase 2):
- ONNX multi-directional broadcasting semantics
- Handles rank mismatches and dimension broadcasts
- Optimized HLS code generation for broadcast patterns
"""

import logging

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM, KernelOp
from brainsmith.dataflow.constraints import CustomConstraint
from brainsmith.dataflow.spec_helpers import (
    add_datatype,
    derive_dim,
    smallest_datatype_for_range,
    sub_datatype,
)
from brainsmith.dataflow.transformation import TransformationResult
from brainsmith.dataflow.types import VALUE_OPTIMIZED, ShapeHierarchy
from brainsmith.registry import kernel

from .operations import BinaryOperations

logger = logging.getLogger(__name__)


# =============================================================================
# Datatype Derivation Helpers
# =============================================================================


def _derive_div_datatype(interfaces, param_getter, model, tensor_name):
    """Compute division output datatype according to FINN/UG1399 rules.

    Division: width = lhs_width if rhs unsigned, else lhs_width + 1
    Signedness: signed if either operand is signed

    Args:
        interfaces: Dict of interface design points
        param_getter: Callable to get kernel parameters
        model: ModelWrapper (unused but required by signature)
        tensor_name: Tensor name (unused but required by signature)

    Returns:
        DataType for division output
    """
    lhs_dt = interfaces["lhs"].datatype
    rhs_dt = interfaces["rhs"].datatype

    lhs_width = lhs_dt.bitwidth()
    signed = any([lhs_dt.signed(), rhs_dt.signed()])

    # Width: lhs_width if rhs unsigned, else lhs_width + 1
    out_width = lhs_width if not rhs_dt.signed() else lhs_width + 1

    return DataType[f"INT{out_width}" if signed else f"UINT{out_width}"]


def _derive_bitwise_datatype(interfaces, param_getter, model, tensor_name):
    """Compute bitwise operation output datatype according to FINN/UG1399 rules.

    Bitwise (AND, OR, XOR): max(lhs_width, rhs_width), preserve signedness

    Args:
        interfaces: Dict of interface design points
        param_getter: Callable to get kernel parameters
        model: ModelWrapper (unused but required by signature)
        tensor_name: Tensor name (unused but required by signature)

    Returns:
        DataType for bitwise operation output
    """
    lhs_dt = interfaces["lhs"].datatype
    rhs_dt = interfaces["rhs"].datatype

    lhs_width = lhs_dt.bitwidth()
    rhs_width = rhs_dt.bitwidth()
    max_width = max(lhs_width, rhs_width)

    signed = any([lhs_dt.signed(), rhs_dt.signed()])

    return DataType[f"INT{max_width}" if signed else f"UINT{max_width}"]


def _elementwise_binary_output_datatype():
    """Polymorphic datatype resolver for ElementwiseBinaryOp.

    Dispatches to appropriate builder based on 'func' nodeattr.
    Supports all 17 binary operations with operation-specific datatype rules.

    Returns:
        Callable resolver function with unified signature
    """

    def resolver(interfaces, param_getter, model, tensor_name):
        func = param_getter("func")
        lhs_dt = interfaces["lhs"].datatype
        rhs_dt = interfaces["rhs"].datatype

        # Check if inputs are integer types
        # For float types, skip bitwidth calculations (matching FINN behavior)
        if not (lhs_dt.is_integer() and rhs_dt.is_integer()):
            # Float type handling:
            # - Comparison operations still return BINARY
            # - Arithmetic/bitwise operations return LHS datatype (float passthrough)
            if func in ("Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"):
                return smallest_datatype_for_range(0, 1)  # Still BINARY for comparisons
            else:
                # Float operations preserve input type
                return lhs_dt

        # Integer type handling (bitwidth-aware derivation):

        # Arithmetic operations (context-aware builders from spec_helpers)
        if func == "Add":
            return add_datatype("lhs", "rhs")(interfaces, param_getter, model, tensor_name)
        elif func == "Sub":
            return sub_datatype("lhs", "rhs")(interfaces, param_getter, model, tensor_name)
        elif func == "Mul":
            # Use FINN's formula: output_width = lhs_width + rhs_width (Vivado HLS UG1399)
            # This follows hardware synthesis rules for multiplication
            lhs_width = lhs_dt.bitwidth()
            rhs_width = rhs_dt.bitwidth()
            signed = lhs_dt.signed() or rhs_dt.signed()
            output_width = lhs_width + rhs_width
            return DataType[f"INT{output_width}" if signed else f"UINT{output_width}"]
        elif func == "Div":
            return _derive_div_datatype(interfaces, param_getter, model, tensor_name)

        # Logical operations → BINARY (0 or 1)
        elif func in ("And", "Or", "Xor"):
            return smallest_datatype_for_range(0, 1)

        # Comparison operations → BINARY (0 or 1)
        elif func in ("Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"):
            return smallest_datatype_for_range(0, 1)

        # Bitwise operations → max width, preserve signedness
        elif func in ("BitwiseAnd", "BitwiseOr", "BitwiseXor"):
            return _derive_bitwise_datatype(interfaces, param_getter, model, tensor_name)

        # BitShift operations → use LHS datatype (shift doesn't change type)
        # Note: "BitShift" is the standard ONNX operation (direction in attribute)
        elif func == "BitShift":
            return lhs_dt

        else:
            raise ValueError(
                f"Unsupported func '{func}' in ElementwiseBinaryOp. "
                f"This should have been caught by schema validation."
            )

    return resolver


# =============================================================================
# Schema Definition
# =============================================================================


def _validate_input_pattern(ctx):
    """Validate input pattern based on weight status.

    Validation rules (derived from mem_mode behavior):
    - If RHS is weight: valid (dynamic_static or MLO dynamic_dynamic)
    - If RHS is not weight: LHS also can't be weight (both streaming)

    Returns:
        None if valid, error message string if invalid
    """
    if "lhs" not in ctx.inputs or "rhs" not in ctx.inputs:
        return "Missing required inputs 'lhs' or 'rhs'"

    lhs = ctx.inputs["lhs"]
    rhs = ctx.inputs["rhs"]

    # Simple validation: if RHS is not a weight, LHS can't be either
    if not rhs.is_weight and lhs.is_weight:
        return "LHS cannot be a weight when RHS is not a weight (invalid pattern)"

    # All valid cases:
    # 1. RHS is weight, LHS is not → dynamic_static
    # 2. RHS is weight (MLO context) → dynamic_dynamic
    # 3. Neither is weight → dynamic_dynamic (both activations)

    return None


ELEMENTWISE_BINARY_SCHEMA = df.KernelSchema(
    name="ElementwiseBinaryOp",
    inputs=[
        df.InputSchema(
            name="lhs",
            block_tiling=[FULL_DIM],  # Process full tensor (no blocking)
            stream_tiling=["PE"],  # PE parallelism on last dimension
            required_layout=None,
        ),
        df.InputSchema(
            name="rhs",
            # Note: Tiling is minimal for backward compatibility with Phase 1 (static)
            # For Phase 2 dynamic+dynamic, HLS backend will create streaming interface
            # based on derived pattern from mem_mode
            block_tiling=[FULL_DIM],  # Full tensor (needed for shape inference)
            stream_tiling=["PE"],  # PE parallelism (used only if dynamic)
            datatype=VALUE_OPTIMIZED,  # Optimize from actual values
            required_layout=None,
            # Memory modes for RHS - static capabilities (what CAN it be if weight)
            mem_modes=frozenset({"embedded", "decoupled", "dynamic"}),  # All possible modes
        ),
    ],
    outputs=[
        df.OutputSchema(
            name="output",
            block_tiling=[FULL_DIM],  # Full tensor
            stream_tiling=[derive_dim("lhs", ShapeHierarchy.STREAM, -1)],  # Match LHS PE
            datatype=_elementwise_binary_output_datatype(),  # Polymorphic dispatch
            required_layout=None,
        )
    ],
    # STRUCTURAL (fixed at inference)
    kernel_params={
        # Operation type: matches ONNX op_type (from operations registry)
        "func": ("s", True, "Add", BinaryOperations.all_operation_names()),
        # Direction for BitShift operations (optional, only used when func="BitShift")
        "direction": (
            "s",
            False,
            "",  # Optional parameter
            {"LEFT", "RIGHT", ""},
        ),
        # NOTE: input_pattern removed - now derived from rhs.mem_mode (single source of truth)
        # Pattern derivation:
        #   - mem_mode in ("embedded", "decoupled"): dynamic_static
        #   - mem_mode == "dynamic" or None: dynamic_dynamic
    },
    # DSE PARAMETERS (explorable resource parameters)
    dse_parameters={
        # RAM style for parameter storage (HLS-specific, only for static inputs)
        "ram_style": df.ParameterSpec(
            name="ram_style", values={"auto", "distributed", "block", "ultra"}, default="auto"
        ),
        # NOTE: mem_mode moved to interface-level (rhs.mem_modes) → generates input1MemType
    },
    constraints=[
        # Pattern-specific validation (dynamic vs static, broadcasting)
        CustomConstraint(
            check_fn=_validate_input_pattern,
            description="Validate input pattern (dynamic_static or dynamic_dynamic)",
        ),
        # Shapes must be broadcastable (ONNX semantics)
        # Note: This is implicitly validated by BroadcastInfo.compute() during HLS codegen
    ],
)


# =============================================================================
# ElementwiseBinaryOp Kernel Implementation
# =============================================================================


@kernel(
    description="Elementwise binary operations (arithmetic, logical, comparison, bitwise) with PE parallelism and broadcasting",
    author="Migrated from AMD FINN by Thomas Keller",
)
class ElementwiseBinaryOp(KernelOp):
    """Hardware kernel for elementwise binary operations.

    Implements a polymorphic kernel supporting 17 binary operations:
    - Arithmetic: Add, Sub, Mul, Div
    - Logical: And, Or, Xor
    - Comparison: Equal, Less, LessOrEqual, Greater, GreaterOrEqual
    - Bitwise: BitwiseAnd, BitwiseOr, BitwiseXor
    - BitShift: BitShiftLeft, BitShiftRight

    Supported Input Patterns:
    - dynamic_static: One streaming input (LHS), one static parameter (RHS)
    - dynamic_dynamic: Both inputs streaming with ONNX broadcasting support

    Broadcasting Features (Phase 2):
    - ONNX multi-directional broadcasting semantics
    - Handles rank mismatches (e.g., [1,64,64,128] + [128])
    - Dimension broadcasts (e.g., [1,64,64,128] + [1,1,1,128])
    - Optimized HLS code generation with conditional reads

    The operation type is determined by the 'func' parameter, and the input
    pattern is determined by the 'input_pattern' parameter, both set during
    inference from the ONNX graph.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> df.KernelSchema:
        return ELEMENTWISE_BINARY_SCHEMA

    # ================================================================
    # Derived Properties (Single Source of Truth)
    # ================================================================

    @property
    def input_pattern(self) -> str | None:
        """Derive input pattern from RHS memory mode (single source of truth).

        Pattern derivation:
        - mem_mode=None: dynamic_dynamic (both activations streaming)
        - mem_mode="embedded"/"decoupled": dynamic_static (static weight)
        - mem_mode="dynamic": dynamic_dynamic (weight from loop/MLO)

        Returns:
            "dynamic_static" or "dynamic_dynamic", or None if not yet configured
        """
        if not hasattr(self, 'design_point') or self.design_point is None:
            return None  # Not yet configured

        rhs_iface = self.design_point.inputs.get("rhs")
        if rhs_iface is None:
            return None

        # Derive from mem_mode
        if rhs_iface.mem_mode in ("embedded", "decoupled"):
            return "dynamic_static"  # Static weight
        elif rhs_iface.mem_mode == "dynamic" or rhs_iface.mem_mode is None:
            return "dynamic_dynamic"  # Streaming (from loop or both activations)
        else:
            raise ValueError(f"Unknown mem_mode: {rhs_iface.mem_mode}")

    # ================================================================
    # ONNX → KernelOp Inference (Unified System)
    # ================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if node can be inferred as ElementwiseBinaryOp.

        Supported Patterns:
        - Phase 1 (dynamic_static): One dynamic, one static input
        - Phase 2 (dynamic_dynamic): Both dynamic inputs with broadcasting

        Args:
            node: ONNX node to check
            model: ModelWrapper for accessing graph info

        Returns:
            True if node can be inferred as this kernel
        """
        from brainsmith.dataflow.inference_helpers import (
            check_shapes_broadcastable,
            find_dynamic_dynamic_pair,
            find_static_dynamic_pair,
        )

        # Must be supported operation
        if node.op_type not in ELEMENTWISE_BINARY_SCHEMA.kernel_params["func"][3]:
            return False

        # Must have exactly 2 inputs
        if len(node.input) != 2:
            return False

        # Check for dynamic+static pattern (Phase 1)
        static_dynamic_pair = find_static_dynamic_pair(node.input, model)
        if static_dynamic_pair is not None:
            # Reject output dequantization (Mul with float output, no successors)
            if node.op_type == "Mul":
                if not model.find_direct_successors(node):
                    out_dtype = model.get_tensor_datatype(node.output[0])
                    if out_dtype.name == "FLOAT32":
                        return False
            return True

        # Check for dynamic+dynamic pattern (Phase 2)
        dynamic_dynamic_pair = find_dynamic_dynamic_pair(node.input, model)
        if dynamic_dynamic_pair is not None:
            # Shapes must be broadcastable
            if not check_shapes_broadcastable(node.input[0], node.input[1], model):
                return False
            return True

        # Neither pattern matched (both static, or other issue)
        return False

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int, kernel_index: int = None
    ) -> TransformationResult:
        """Infer ElementwiseBinaryOp from ONNX binary operation.

        Detects input pattern and creates appropriate HW node:
        - Phase 1 (dynamic_static): Reorders to (dynamic, static)
        - Phase 2 (dynamic_dynamic): Preserves order, validates broadcasting

        Args:
            node: ONNX node to transform
            model: ModelWrapper for accessing graph info
            insert_index: Index where to insert new node
            kernel_index: Sequential index for this kernel type (for naming)

        Returns:
            TransformationResult with new HW node
        """
        from brainsmith.dataflow import lift_scalar_to_rank1
        from brainsmith.dataflow.inference_helpers import (
            find_dynamic_dynamic_pair,
            find_static_dynamic_pair,
            get_broadcast_info,
        )

        # IMPORTANT: Normalize scalar inputs before pattern detection and validation
        # ONNX semantics: [] (scalar) broadcasts identically to [1] in all contexts
        # This lifting is required because our schema system uses template-based tiling
        # (e.g., block_tiling=[FULL_DIM]) which expects rank ≥ 1 tensors.
        # Safe to mutate model here - QONNX transforms operate on deep copies by default.
        for inp in node.input:
            lift_scalar_to_rank1(inp, model)

        # Try dynamic+static pattern first (Phase 1)
        static_dynamic_pair = find_static_dynamic_pair(node.input, model)
        if static_dynamic_pair is not None:
            lhs_input, rhs_input = static_dynamic_pair  # (dynamic, static)
            # Pattern will be derived from mem_mode: "dynamic_static"

        else:
            # Try dynamic+dynamic pattern (Phase 2)
            dynamic_dynamic_pair = find_dynamic_dynamic_pair(node.input, model)
            if dynamic_dynamic_pair is not None:
                lhs_input, rhs_input = dynamic_dynamic_pair  # Both dynamic
                # Pattern will be derived from mem_mode: "dynamic_dynamic"

                # Log broadcasting information for debugging
                broadcast_info = get_broadcast_info(lhs_input, rhs_input, model)
                if broadcast_info and broadcast_info.has_broadcast:
                    logger.info(
                        f"ElementwiseBinaryOp {node.name}: Broadcasting detected - "
                        f"LHS{broadcast_info.lhs_shape} + RHS{broadcast_info.rhs_shape} "
                        f"→ {broadcast_info.output_shape}"
                    )
            else:
                raise ValueError(
                    f"Node {node.name} doesn't match any ElementwiseBinaryOp pattern. "
                    f"Expected either (dynamic, static) or (dynamic, dynamic) inputs."
                )

        # Create ElementwiseBinaryOp node with sequential naming
        # NOTE: input_pattern removed - now derived from rhs.mem_mode during design space building
        node_name = f"ElementwiseBinaryOp_{kernel_index}" if kernel_index is not None else node.name
        hw_node = helper.make_node(
            "ElementwiseBinaryOp",
            inputs=[lhs_input, rhs_input],
            outputs=node.output,
            name=node_name,
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            # Kernel parameters
            func=node.op_type,
            # input_pattern removed - derived from mem_mode
        )

        # Copy direction attribute for BitShift operations
        if node.op_type == "BitShift":
            from onnx import helper as onnx_helper

            direction_attr = None
            for attr in node.attribute:
                if attr.name == "direction":
                    direction_attr = onnx_helper.get_attribute_value(attr)
                    break
            if direction_attr:
                hw_node.attribute.append(onnx_helper.make_attribute("direction", direction_attr))
            else:
                raise ValueError(
                    f"BitShift node {node.name} missing required 'direction' attribute"
                )

        # Mark RHS as weight if it's an initializer (for mem_mode parameter creation)
        # Attribute presence indicates weight; absence indicates pure streaming input
        rhs_input = hw_node.input[1]
        if model.get_initializer(rhs_input) is not None:
            hw_node.attribute.append(helper.make_attribute("input1MemType", "embedded"))

        # Copy metadata_props (e.g., PyTorch name scopes for loop rolling)
        # metadata_props is a protobuf RepeatedCompositeFieldContainer
        if hasattr(node, 'metadata_props') and len(node.metadata_props) > 0:
            for entry in node.metadata_props:
                new_entry = hw_node.metadata_props.add()
                new_entry.key = entry.key
                new_entry.value = entry.value

        # Return transformation result
        return TransformationResult(
            nodes_to_remove=[node],
            nodes_to_insert=[hw_node],
        )

    # ================================================================
    # Execution Support (For cppsim compatibility)
    # ================================================================

    @property
    def npy_op(self):
        """NumPy operation for execute_node() simulation.

        Maps 'func' parameter to corresponding NumPy ufunc.
        For BitShift operations, also considers 'direction' attribute.

        Returns:
            NumPy ufunc for the operation
        """
        import numpy as np

        from .operations import BinaryOperations

        func = self.get_nodeattr("func")

        # Special handling for BitShift with direction attribute
        if func == "BitShift":
            direction = self.get_nodeattr("direction")
            if direction == "LEFT":
                return np.left_shift
            elif direction == "RIGHT":
                return np.right_shift
            else:
                raise ValueError(f"Invalid BitShift direction: {direction} (must be LEFT or RIGHT)")

        return BinaryOperations.get_npy_op(func)

    @property
    def cpp_op(self):
        """C++ operation template for HLS code generation.

        Maps 'func' parameter to C++ expression template.
        Templates use {0} for LHS and {1} for RHS.
        For BitShift operations, also considers 'direction' attribute.

        Returns:
            C++ expression template string
        """
        from .operations import BinaryOperations

        func = self.get_nodeattr("func")

        # Special handling for BitShift with direction attribute
        if func == "BitShift":
            direction = self.get_nodeattr("direction")
            if direction == "LEFT":
                return "({0} << {1})"
            elif direction == "RIGHT":
                return "({0} >> {1})"
            else:
                raise ValueError(f"Invalid BitShift direction: {direction} (must be LEFT or RIGHT)")

        return BinaryOperations.get_cpp_template(func)

    # ================================================================
    # Configuration Validation (Phase 3c)
    # ================================================================

    def validate_configuration(self, model=None):
        """Validate kernel configuration before building.

        Performs production-grade validation checks:
        - PE divisibility
        - Broadcast compatibility
        - Operation support
        - Datatype compatibility
        - Edge case warnings (division by zero, overflow)

        Args:
            model: Optional ModelWrapper for accessing initializers

        Raises:
            ValueError: With helpful error message if invalid configuration

        Warnings:
            RuntimeWarning: For edge cases (division by zero, overflow risk)
        """
        # Validate PE divisibility
        self._validate_pe_divisibility()

        # Validate broadcast compatibility
        self._validate_broadcast_compatibility()

        # Validate operation support
        self._validate_operation_support()

        # Note: Datatype compatibility is enforced by schema's DatatypeInteger
        # constraint, so no runtime validation needed here.

        # Check edge cases (warnings, not errors)
        self._handle_zero_division(model)
        self._check_overflow_risk()

    def _validate_pe_divisibility(self):
        """Validate PE evenly divides channel dimension.

        Raises:
            ValueError: If PE doesn't divide channels, with suggested alternatives
        """
        if not hasattr(self, "design_point") or self.design_point is None:
            # Skip if design_point not initialized yet
            return

        lhs_shape = self.design_point.inputs["lhs"].tensor_shape
        pe = self.design_point.inputs["lhs"].stream_shape[-1]
        num_channels = lhs_shape[-1]

        if num_channels % pe != 0:
            valid_pe_values = self._suggest_pe_values(num_channels)
            raise ValueError(
                f"{self.onnx_node.name}: PE={pe} must evenly divide "
                f"channel dimension {num_channels}.\n"
                f"  Valid PE values for {num_channels} channels: "
                f"{', '.join(map(str, valid_pe_values))}\n"
                f"  Example: Set PE={valid_pe_values[len(valid_pe_values)//2]} "
                f"for moderate parallelism."
            )

    def _suggest_pe_values(self, channels):
        """Suggest valid PE values for given channel count.

        Args:
            channels: Number of channels

        Returns:
            List of valid PE values (all divisors of channels, sorted)
        """
        # Find all divisors by checking up to sqrt(channels)
        divisors = []
        for i in range(1, int(channels**0.5) + 1):
            if channels % i == 0:
                divisors.append(i)
                if i != channels // i:  # Avoid duplicates for perfect squares
                    divisors.append(channels // i)
        return sorted(divisors)

    def _validate_broadcast_compatibility(self):
        """Validate input shapes are broadcastable (for dynamic_dynamic pattern).

        Raises:
            ValueError: If shapes not broadcastable, with examples
        """
        # Use property to derive pattern from mem_mode
        input_pattern = self.input_pattern

        if input_pattern != "dynamic_dynamic":
            # Only validate for dynamic_dynamic pattern (both inputs streaming)
            return

        if not hasattr(self, "design_point") or self.design_point is None:
            # Skip if design_point not initialized yet
            return

        lhs_shape = self.design_point.inputs["lhs"].tensor_shape
        rhs_shape = self.design_point.inputs["rhs"].tensor_shape

        try:
            np.broadcast_shapes(lhs_shape, rhs_shape)
        except ValueError as e:
            raise ValueError(
                f"{self.onnx_node.name}: Input shapes not broadcastable.\n"
                f"  LHS shape: {lhs_shape}\n"
                f"  RHS shape: {rhs_shape}\n"
                f"  ONNX broadcasting rules:\n"
                f"    - Shapes align from the right\n"
                f"    - Dimensions must be equal OR one must be 1\n"
                f"  Examples of valid broadcasts:\n"
                f"    [1,64,64,128] + [128] → [1,64,64,128] (channel broadcast)\n"
                f"    [1,64,64,128] + [1,1,1,128] → [1,64,64,128] (spatial broadcast)\n"
                f"    [1,64,1,128] + [1,1,64,1] → [1,64,64,128] (bidirectional)\n"
                f"  Original error: {e}"
            )

    def _validate_operation_support(self):
        """Validate operation is supported.

        Raises:
            ValueError: If operation not supported
        """
        operation = self.get_nodeattr("func")
        # Get supported ops from schema
        schema_ops = ELEMENTWISE_BINARY_SCHEMA.kernel_params["func"][3]

        if operation not in schema_ops:
            supported_list = sorted(schema_ops)
            raise ValueError(
                f"{self.onnx_node.name}: Unsupported operation '{operation}'.\n"
                f"  Supported operations: {', '.join(supported_list)}\n"
                f"  Note: This check is redundant with schema validation."
            )

    def _handle_zero_division(self, model=None):
        """Check for potential division by zero in Div operation.

        Args:
            model: Optional ModelWrapper for accessing initializers

        Warnings:
            RuntimeWarning: If RHS initializer contains zeros
        """
        if self.get_nodeattr("func") != "Div":
            return

        # Need model to check initializers
        if model is None:
            return

        # Check RHS for zeros (only if it's an initializer)
        rhs_name = self.onnx_node.input[1]

        if rhs_name in [init.name for init in model.graph.initializer]:
            rhs_data = model.get_initializer(rhs_name)
            if rhs_data is not None and np.any(rhs_data == 0):
                import warnings

                warnings.warn(
                    f"{self.onnx_node.name}: RHS initializer contains zeros. "
                    f"Division by zero will produce undefined results.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _check_overflow_risk(self):
        """Check if operation might overflow output datatype.

        Warnings:
            RuntimeWarning: If output type might be too small
        """
        operation = self.get_nodeattr("func")
        if operation not in {"Add", "Mul"}:
            return  # Sub/Div less likely to overflow

        if not hasattr(self, "design_point") or self.design_point is None:
            return

        lhs_dt = self.design_point.inputs["lhs"].datatype
        out_dt = self.design_point.outputs["output"].datatype

        # For Add: need at least input_width + 1 bit
        # For Mul: need at least 2 * input_width bits
        required_bits = lhs_dt.bitwidth() + 1 if operation == "Add" else 2 * lhs_dt.bitwidth()

        if out_dt.bitwidth() < required_bits:
            import warnings

            warnings.warn(
                f"{self.onnx_node.name}: Output datatype {out_dt} may be too "
                f"small for {operation} operation on {lhs_dt} inputs.\n"
                f"  Input bitwidth: {lhs_dt.bitwidth()}\n"
                f"  Output bitwidth: {out_dt.bitwidth()}\n"
                f"  Recommended minimum: {required_bits} bits\n"
                f"  Consider using a wider output type or adding saturation.",
                RuntimeWarning,
                stacklevel=2,
            )

    def execute_node(self, context, graph):
        """Execute node in Python simulation.

        Applies the elementwise operation with broadcasting semantics.

        Args:
            context: Execution context dict (tensor_name -> numpy array)
            graph: ONNX graph (for metadata)
        """
        # Ensure initialized (QONNX executor creates fresh instances)
        self._ensure_initialized_for_execution(graph)

        node = self.onnx_node

        # Get inputs from context
        lhs = context[node.input[0]]
        rhs = context[node.input[1]]

        # Get datatypes
        lhs_dtype = self.design_point.inputs["lhs"].datatype
        rhs_dtype = self.design_point.inputs["rhs"].datatype
        operation = self.get_nodeattr("func")

        # Convert to appropriate types for NumPy
        # Use int64 for integer types to avoid overflow in intermediate computation
        lhs = lhs.astype(np.int64) if lhs_dtype.is_integer() else lhs
        rhs = rhs.astype(np.int64) if rhs_dtype.is_integer() else rhs

        # Apply operation with broadcasting
        # Special handling for division: use truncating division for integer types to match hardware
        if operation == "Div" and lhs_dtype.is_integer() and rhs_dtype.is_integer():
            # Integer division: truncate toward zero (C/C++ semantics), not floor division
            # This matches hardware implementation in HLS/RTL
            out = np.trunc(np.divide(lhs, rhs)).astype(np.int64)
        else:
            # All other operations: use standard NumPy operation
            out = self.npy_op(lhs, rhs)

        # Store result (as float32 container, QONNX convention)
        context[node.output[0]] = out.astype(np.float32)

    # ================================================================
    # MLO Loop Body Adaptation
    # ================================================================

    def adapt_for_loop_body(self, loop_signature):
        """Adapt ElementwiseBinaryOp for use in FINNLoop body.

        Forces RHS memory mode to "dynamic" when weights are streamed from loop level.
        Only modifies the attribute if:
        1. RHS is marked as weight (attribute exists from InferKernel)
        2. Loop signature indicates input is PARAMETER (streamed per iteration)

        Args:
            loop_signature: List of LoopBodyInputType values for each input
        """
        from qonnx.util.basic import get_by_name

        # Check if RHS is marked as weight
        attr = get_by_name(self.onnx_node.attribute, "input1MemType")
        if attr is None:
            return  # Not a weight, nothing to adapt

        # Check if loop signature indicates this input is streamed as parameter
        if loop_signature and len(loop_signature) > 1:
            from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType

            if loop_signature[1] == LoopBodyInputType.PARAMETER:
                self.set_nodeattr("input1MemType", "dynamic")
                logger.debug(f"{self.onnx_node.name}: Forced input1MemType=dynamic for MLO")

    # ================================================================
    # ONNX Shape Compatibility
    # ================================================================

    def make_shape_compatible_op(self, model):
        """Make ElementwiseBinaryOp compatible with ONNX shape inference.

        Returns equivalent ONNX op that has standard shape inference.
        For binary operations, use Add which supports broadcasting.

        Args:
            model: ModelWrapper (unused but required by signature)

        Returns:
            ONNX NodeProto compatible with shape inference
        """
        # Use Add for shape inference (supports broadcasting)
        return helper.make_node("Add", self.onnx_node.input, self.onnx_node.output)
