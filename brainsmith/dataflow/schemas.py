############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Consolidated schema definitions for dataflow kernels.

Schemas define STRUCTURE, not STORAGE. They specify:
- Interface definitions (inputs, outputs)
- Tiling templates (block_tiling, stream_tiling)
- Validation rules (constraints, relationships)

Schemas are stateless - they never define what gets stored in nodeattrs.
Storage decisions are implementation details of operator classes.

Key classes:
- InterfaceSchema: Base class for input/output interfaces
- InputSchema: Schema for input interfaces
- OutputSchema: Schema for output interfaces
- KernelSchema: Complete kernel definition with inputs and outputs
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .types import FULL_DIM, FULL_SHAPE

if TYPE_CHECKING:
    from .builder import BuildContext
    from .constraints import Constraint

logger = logging.getLogger(__name__)

# Type aliases for better clarity
TilingSpec = (
    Sequence[int | str | type(FULL_DIM) | Callable]  # List of DimSpecs
    | type(FULL_SHAPE)  # Bare sentinel (rank-agnostic)
)


# ============================================================================
# DSE Dimension Type
# ============================================================================


@dataclass(frozen=True)
class ParameterSpec:
    """Explorable parameter in design space.

    Represents resource allocation or implementation choices that can be
    explored during DSE (ram_style, res_type, mem_mode, etc.).

    Does NOT include tiling dimensions (PE, SIMD) - those are auto-extracted
    from stream_tiling templates with valid values computed from factoring.

    **Container Type Convention (Ordered vs Discrete):**

    The container type determines how the dimension is treated during DSE:

    - **list/tuple → OrderedParameter** (ordered sequences with navigation)
        - Supports min/max access, step_up/step_down, percentage-based indexing
        - Values are sorted automatically
        - Examples: depth=[128, 256, 512], num_layers=[1, 2, 4, 8]

    - **set/frozenset → Discrete** (unordered categories)
        - Membership testing only, no navigation
        - Order doesn't matter
        - Examples: ram_style={"distributed", "block"}, res_type={"lut", "dsp"}

    **Type Declaration (Hybrid Approach):**

    - **Literal values**: Type inferred from first value (optional to specify)
    - **Callable values**: Type MUST be explicitly specified

    Attributes:
        name: Dimension name (e.g., "ram_style", "depth")
        values: Valid values for this dimension
            - list/tuple: Ordered sequence (enables navigation methods)
            - set/frozenset: Discrete categories (membership only)
            - Callable: Computed from BuildContext (for context-dependent values)
        type: Value type ("int" or "string")
            - Required for callable values
            - Optional for literal values (inferred from first value)
            - Validated against values if both provided
        default: Default value (None = auto-select: min for ordered, first for discrete)

    Examples:
        >>> # Ordered parameter - type inferred
        >>> ParameterSpec("depth", [128, 256, 512, 1024], default=256)

        >>> # Discrete parameter - type inferred
        >>> ParameterSpec("ram_style", {"distributed", "block"}, default="distributed")

        >>> # Callable parameter - type required
        >>> ParameterSpec("depth", lambda ctx: compute_depths(ctx), type="int", default=256)

        >>> # Explicit type for documentation (optional)
        >>> ParameterSpec("mode", {"fast", "accurate"}, type="string")

    Validation:
        - Callable values without type → ValueError
        - Type mismatch with literal values → ValueError
        - Invalid type (not "int" or "string") → ValueError

    Note:
        Tiling dimensions (PE, SIMD) are ALWAYS ordered (auto-wrapped in OrderedParameter)
        since they're computed as divisors (naturally ordered sequences).
    """

    name: str
    values: set[int | str] | Callable[["BuildContext"], set[int | str]]
    type: Literal["int", "string"] | None = None
    default: int | str | None = None

    def __post_init__(self):
        """Validate type specification against values."""
        # Callable values MUST specify type
        if callable(self.values):
            if self.type is None:
                raise ValueError(
                    f"ParameterSpec '{self.name}': Callable values require explicit type declaration. "
                    f"Specify type='int' or type='string'."
                )
            if self.type not in ("int", "string"):
                raise ValueError(
                    f"ParameterSpec '{self.name}': Invalid type '{self.type}'. "
                    f"Must be 'int' or 'string'."
                )
            return  # Cannot validate callable values without context

        # Validate type matches literal values (if type specified)
        if self.type is not None:
            if self.type not in ("int", "string"):
                raise ValueError(
                    f"ParameterSpec '{self.name}': Invalid type '{self.type}'. "
                    f"Must be 'int' or 'string'."
                )

            first_val = next(iter(self.values))
            expected_type = int if self.type == "int" else str

            if not isinstance(first_val, expected_type):
                actual_type = type(first_val).__name__
                raise ValueError(
                    f"ParameterSpec '{self.name}': Type mismatch. "
                    f"Declared type='{self.type}' but values contain {actual_type}. "
                    f"First value: {first_val!r}"
                )


# ============================================================================
# Helper Functions
# ============================================================================


def _infer_nodeattr_type(param_spec: ParameterSpec) -> tuple[str, bool, int | str]:
    """Infer FINN nodeattr type specification from ParameterSpec.

    Uses explicit type if provided, otherwise infers from first value.
    This function handles the translation from ParameterSpec (DSE model)
    to FINN nodeattr format (type_code, required, default).

    Args:
        param_spec: ParameterSpec to infer type from

    Returns:
        Tuple of (type_code, required, default_value) where:
        - type_code: "i" for int, "s" for string
        - required: Always False for DSE parameters
        - default_value: Explicit default or inferred minimum/first

    Note:
        For callable values with explicit type, we use that type.
        Default falls back to 1 for int (safe minimum) and first sorted
        value for string.

    Examples:
        >>> spec = ParameterSpec("depth", [128, 256, 512], default=256)
        >>> _infer_nodeattr_type(spec)
        ("i", False, 256)

        >>> spec = ParameterSpec("mode", {"fast", "slow"})
        >>> _infer_nodeattr_type(spec)
        ("s", False, "fast")  # sorted first

        >>> spec = ParameterSpec("depth", lambda ctx: [128, 256], type="int", default=128)
        >>> _infer_nodeattr_type(spec)
        ("i", False, 128)
    """
    # Use explicit type if provided (required for callables)
    if param_spec.type is not None:
        type_code = "i" if param_spec.type == "int" else "s"
        default = (
            param_spec.default
            if param_spec.default is not None
            else (1 if param_spec.type == "int" else "")
        )
        return (type_code, False, default)

    # Infer from first value (literal values only)
    if callable(param_spec.values):
        # Should not reach here - ParameterSpec.__post_init__ enforces type for callables
        raise ValueError(
            f"ParameterSpec '{param_spec.name}': Cannot infer type from callable without explicit type declaration."
        )

    first_val = next(iter(param_spec.values))

    if isinstance(first_val, int):
        # Integer parameter: use minimum as default
        default = param_spec.default if param_spec.default is not None else min(param_spec.values)
        return ("i", False, default)
    else:
        # String parameter: use sorted first as default
        default = (
            param_spec.default if param_spec.default is not None else sorted(param_spec.values)[0]
        )
        return ("s", False, default)


def _extract_tiling_params(
    block_tiling: TilingSpec | None, stream_tiling: TilingSpec | None
) -> list[str]:
    """Extract unique string parameters from tiling specs."""
    params = set()
    for spec in (block_tiling, stream_tiling):
        if spec is None:
            continue
        # Skip FULL_SHAPE sentinel (has no parameters - expands to FULL_DIM)
        if spec is FULL_SHAPE:
            continue
        # Extract string parameters from list-based specs
        params.update(dim for dim in spec if isinstance(dim, str))
    return list(params)


# ============================================================================
# SCHEMA TYPES (Unified System)
# ============================================================================


@dataclass(frozen=True)
class InputSchema:
    """Input interface specification.

    Defines input structure (tiling) and requirements (layout, datatype).

    Attributes:
        name: Interface name (e.g., "input", "input0")
        block_tiling: Block tiling specification (e.g., [FULL_DIM, FULL_DIM])
        stream_tiling: Stream tiling specification (e.g., ["SIMD"], [1, 1, 1, "PE"])
        datatype: Datatype spec (None to use from ONNX, or DatatypeSpec union type to derive/optimize)
        required_layout: Expected input layout (e.g., "NHWC", "NCHW"), None if no requirement
        mem_modes: Memory mode options for weight inputs (frozenset or callable returning frozenset).
                   Valid modes: "embedded" (compile-time constant), "decoupled" (separate memory),
                   "dynamic"/"external" (streaming). Generates input<idx>MemType DSE parameter.
    """

    # Identity
    name: str

    # Structure
    block_tiling: TilingSpec | None = None
    stream_tiling: TilingSpec | None = None
    datatype: Any | None = None  # DatatypeSpec union type

    # Transformation requirements (NEW - embedded in interface)
    required_layout: str | None = None

    # Memory mode specification for weight inputs
    mem_modes: frozenset[str] | Callable | None = None

    def __post_init__(self):
        """Validate interface requirements."""
        if self.required_layout and self.required_layout not in {"NCHW", "NHWC"}:
            raise ValueError(
                f"Invalid required_layout '{self.required_layout}' for input '{self.name}'. "
                f"Must be 'NCHW' or 'NHWC'."
            )

        # Validate mem_modes if specified
        if self.mem_modes is not None and not callable(self.mem_modes):
            VALID_MEM_MODES = {"embedded", "decoupled", "dynamic", "external"}
            if not isinstance(self.mem_modes, frozenset):
                raise TypeError(
                    f"mem_modes for input '{self.name}' must be frozenset or callable, "
                    f"got {type(self.mem_modes).__name__}"
                )
            invalid = self.mem_modes - VALID_MEM_MODES
            if invalid:
                raise ValueError(
                    f"Invalid mem_modes {invalid} for input '{self.name}'. "
                    f"Valid modes: {VALID_MEM_MODES}"
                )

    @property
    def tiling_attrs(self) -> list[str]:
        """Extract unique template parameter names from tiling specs."""
        return _extract_tiling_params(self.block_tiling, self.stream_tiling)


@dataclass(frozen=True)
class OutputSchema:
    """Output interface specification.

    Defines output structure (tiling), datatype derivation, and layout requirements.

    Attributes:
        name: Interface name (e.g., "output", "output0")
        block_tiling: Block tiling specification
        stream_tiling: Stream tiling specification
        datatype: Datatype spec (None to use from ONNX, or DatatypeSpec union type to derive)
        required_layout: Expected output layout (e.g., "NHWC"), None if no requirement
        preserves_input_layout: Whether output preserves first input's layout (default True)
    """

    # Identity
    name: str

    # Structure
    block_tiling: TilingSpec | None = None
    stream_tiling: TilingSpec | None = None
    datatype: Any | None = None  # DatatypeSpec union type

    # Transformation requirements (NEW)
    required_layout: str | None = None
    preserves_input_layout: bool = True  # Most kernels preserve layout

    def __post_init__(self):
        """Validate interface requirements."""
        if self.required_layout and self.required_layout not in {"NCHW", "NHWC"}:
            raise ValueError(
                f"Invalid required_layout '{self.required_layout}' for output '{self.name}'. "
                f"Must be 'NCHW' or 'NHWC'."
            )

    @property
    def tiling_attrs(self) -> list[str]:
        """Extract unique template parameter names from tiling specs."""
        return _extract_tiling_params(self.block_tiling, self.stream_tiling)


@dataclass
class KernelSchema:
    """Kernel specification defining structure and validation.

    Combines interface definitions, validation constraints, and design space
    parameters. Defines structure only - shapes come from ONNX context,
    execution logic lives in KernelOp.

    Attributes:
        name: Kernel name
        inputs: Input interface schemas
        outputs: Output interface schemas
        internal_datatypes: Internal datatype derivation specs (e.g., accumulator)
        kernel_params: Kernel-specific parameters (e.g., epsilon, algorithm)
        dse_parameters: Explorable resource/implementation parameters (e.g., ram_style)
        constraints: Validation constraints (datatype, shape, ONNX requirements)
        attribute_mapping: Map ONNX attributes to kernel parameters
    """

    # ============= IDENTITY =============
    name: str

    # ============= STRUCTURE =============
    inputs: list[InputSchema] = field(default_factory=list)
    outputs: list[OutputSchema] = field(default_factory=list)
    internal_datatypes: dict[str, Any] = field(default_factory=dict)  # DatatypeSpec union type
    kernel_params: dict[str, tuple] = field(default_factory=dict)

    # ============= DSE PARAMETERS =============
    dse_parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    """Explorable resource/implementation parameters (ram_style, res_type, etc.).

    Tiling parameters (PE, SIMD) NOT declared here - auto-extracted from
    stream_tiling templates with defaults computed from factoring.

    Example: {"ram_style": ParameterSpec("ram_style", {"distributed", "block"}, "distributed")}
    """

    # ============= VALIDATION =============
    constraints: list["Constraint"] = field(default_factory=list)

    # ============= TRANSFORMATION =============
    attribute_mapping: dict[str, str] = field(default_factory=dict)
    """Map ONNX attributes to kernel parameters.

    Example: {"epsilon": "epsilon", "axis": "normalized_axis"}
    """

    def __post_init__(self):
        """Validate schema structure and transformation consistency."""
        self.validate()

        # Validate transformation fields
        self._validate_transformation_fields()

    def validate(self) -> None:
        """Validate the schema structure."""

        # Create sets directly (no intermediate lists)
        input_names = {inp.name for inp in self.inputs}
        output_names = {out.name for out in self.outputs}

        # Check for duplicates (comparing lengths to original counts)
        if len(input_names) != len(self.inputs):
            raise ValueError(f"Duplicate input names in kernel '{self.name}'")
        if len(output_names) != len(self.outputs):
            raise ValueError(f"Duplicate output names in kernel '{self.name}'")

        # Check for conflicts between inputs and outputs
        conflicts = input_names & output_names
        if conflicts:
            raise ValueError(
                f"Interface names must be unique across inputs and outputs in kernel '{self.name}'. "
                f"Duplicate names: {', '.join(sorted(conflicts))}"
            )

        # Check internal datatypes (use set operation instead of loop)
        all_interface_names = input_names | output_names
        internal_conflicts = set(self.internal_datatypes) & all_interface_names
        if internal_conflicts:
            raise ValueError(
                f"Internal datatypes conflict with interface names in kernel '{self.name}': "
                f"{', '.join(sorted(internal_conflicts))}"
            )

    def _validate_transformation_fields(self) -> None:
        """Validate transformation-related fields are consistent."""
        # Validate attribute_mapping references kernel_params
        for hw_param in self.attribute_mapping.values():
            if hw_param not in self.kernel_params:
                raise ValueError(
                    f"attribute_mapping maps to '{hw_param}' but it's not in kernel_params. "
                    f"Available params: {list(self.kernel_params.keys())}"
                )

        # Validate dse_parameters have unique names
        for param_name in self.dse_parameters.keys():
            if param_name in self.kernel_params:
                raise ValueError(
                    f"DSE parameter '{param_name}' conflicts with kernel_param. "
                    f"DSE parameters must have unique names."
                )

    def build_nodeattr_registry(self) -> dict[str, tuple]:
        """Build nodeattr registry from schema definition.

        Schemas define STRUCTURE, not STORAGE. Generates persistence layer
        from structural schema, returning only attributes that need persistence:
        - Datatypes (for interfaces and internals)
        - Tiling parameters (SIMD, PE, etc.) - auto-extracted from stream_tiling
        - DSE parameters (ram_style, res_type, etc.) - from dse_parameters
        - Kernel-specific parameters (epsilon, algorithm, etc.) - from kernel_params

        Shapes are NEVER stored in nodeattrs. They are either:
        - Tensor shapes: extracted from ModelWrapper (ONNX graph)
        - Block/stream shapes: computed from schema templates

        Returns:
            Dict mapping nodeattr name to (type, required, default_value)
            Format: {"attrName": ("i"|"s"|"f", True|False, default)}
        """
        attrs = {}

        # Datatypes
        for i in range(len(self.inputs)):
            attrs[f"input{i}Datatype"] = ("s", False, "")

        for i in range(len(self.outputs)):
            attrs[f"output{i}Datatype"] = ("s", False, "")

        for internal_name in self.internal_datatypes.keys():
            attrs[f"{internal_name}Datatype"] = ("s", False, "")

        # Tiling parameters (PE, SIMD, etc.) - auto-extracted
        template_params = self._extract_template_params()
        for param in template_params:
            attrs[param] = ("i", False, 1)  # Default 1, will be computed from factoring

        # Memory mode parameters (input<idx>MemType) - auto-extracted from mem_modes
        for idx, inp in enumerate(self.inputs):
            if inp.mem_modes is not None:
                # Add input<idx>MemType as a string parameter
                attrs[f"input{idx}MemType"] = ("s", False, "embedded")

        # DSE parameters (resource parameters)
        for param_name, param_spec in self.dse_parameters.items():
            attrs[param_name] = _infer_nodeattr_type(param_spec)

        # Kernel-specific parameters (structural)
        attrs.update(self.kernel_params)

        return attrs

    def _extract_template_params(self) -> set:
        """Extract unique template parameter names from tiling specs.

        Returns:
            Set of template parameter names (strings found in tiling specs)
        """
        params = set()
        for interface in self.inputs + self.outputs:
            params.update(interface.tiling_attrs)
        return params

    def get_structural_nodeattrs(self) -> set:
        """Get nodeattrs that affect design space (rebuild if changed).

        Structural nodeattrs are those whose changes require rebuilding
        the entire KernelDesignSpace (not just reconfiguration).

        These include:
        - All datatypes (input, output, internal): Affect internal datatype
          derivation (e.g., accumulator width depends on input datatype)
        - Parameters in block_tiling (rare): Affect block shape computation

        Returns:
            Set of structural nodeattr names

        Example:
            >>> schema.get_structural_nodeattrs()
            {'input0Datatype', 'output0Datatype', 'accumulatorDatatype'}
        """
        structural = set()

        # All datatypes are structural
        for i in range(len(self.inputs)):
            structural.add(f"input{i}Datatype")
        for i in range(len(self.outputs)):
            structural.add(f"output{i}Datatype")
        for internal_name in self.internal_datatypes.keys():
            structural.add(f"{internal_name}Datatype")

        # Parameters in block_tiling are structural (rare)
        for inp in self.inputs:
            if inp.block_tiling and inp.block_tiling is not FULL_SHAPE:
                for elem in inp.block_tiling:
                    if isinstance(elem, str):
                        structural.add(elem)
        for out in self.outputs:
            if out.block_tiling and out.block_tiling is not FULL_SHAPE:
                for elem in out.block_tiling:
                    if isinstance(elem, str):
                        structural.add(elem)

        return structural

    def get_optimization_nodeattrs(self) -> set:
        """Get nodeattrs that affect optimization (re-explore if changed).

        Optimization nodeattrs are those whose changes only require
        re-exploring the design space (trying different stream shapes),
        not rebuilding the entire design space.

        These include:
        - Parallelization parameters (SIMD, PE, MW, MH, etc.): Appear in
          stream_tiling templates and determine stream shapes during DSE

        Returns:
            Set of optimization nodeattr names

        Example:
            >>> schema.get_optimization_nodeattrs()
            {'SIMD', 'PE'}
        """
        # Parameters in stream_tiling affect optimization
        return self._extract_template_params()
