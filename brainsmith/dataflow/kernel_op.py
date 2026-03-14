############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Kernel operator base class.

EXECUTION COMPATIBILITY NOTES
==============================

KernelOp uses lazy initialization for performance. When executed via QONNX's
execute_onnx(), the executor creates fresh instances per node, losing cached
design_space.

If your kernel implements execute_node(), call _ensure_initialized_for_execution(graph)
at the start:

    def execute_node(self, context, graph):
        # Ensure design_space initialized (QONNX executor creates fresh instances)
        self._ensure_initialized_for_execution(graph)

        # Now safe to access design_point (regenerates from design_space)
        dtype = self.design_point.inputs["input"].datatype
        # ... rest of execution logic

This reconstructs ModelWrapper from the graph parameter to initialize design_space
on demand. The overhead is minimal (~1ms) and only occurs when design_space is None
(fresh instances). Design points regenerate automatically from nodeattrs.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Union

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from onnx import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from .builder import BuildContext, DesignSpaceBuilder
from .dse_models import KernelDesignPoint, KernelDesignSpace
from .ordered_parameter import OrderedParameter
from .schemas import KernelSchema
from .transformation import TransformationResult

logger = logging.getLogger(__name__)


class KernelOpError(Exception):
    """Exception raised by kernel operators with node context.

    Attributes:
        node: ONNX node that caused the error
        message: Error message
    """

    def __init__(self, node, message):
        self.node = node
        super().__init__(f"{node.name}: {message}")


class KernelOp(HWCustomOp, ABC):
    """Kernel operator base class.

    Shapes extracted from ModelWrapper context, never stored in nodeattrs.
    Subclasses implement build_schema() to construct their KernelSchema.

    Caching Strategy:
        - design_space: Cached (expensive to build, invalidated on structural changes)
        - design_point: Regenerated from nodeattrs (guarantees consistency)

    For execution compatibility notes, see module docstring.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        # Build and freeze schema from node structure
        self.kernel_schema = self.build_schema(onnx_node, model=None)

        # Design space caching for DSE performance
        # Design points are regenerated on each access to ensure consistency with nodeattrs
        self._design_space: "KernelDesignSpace" | None = (
            None  # Cached, call invalidate() to rebuild
        )

    @classmethod
    @abstractmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> KernelSchema:
        """Build kernel schema from ONNX node.

        Polymorphic method that handles both static and dynamic schemas:
        - Static schemas: return constant, ignore parameters
        - Dynamic schemas: inspect node structure to build schema

        Called in two contexts:
        1. During __init__: model=None (schema built for instance)
        2. During can_infer_from(): model provided (schema built for validation)

        Args:
            node: ONNX node (provides inputs, outputs, attributes)
            model: Optional ModelWrapper (provides shapes, datatypes for validation context)

        Returns:
            KernelSchema defining kernel structure

        Example (static schema):
            @classmethod
            def build_schema(cls, node, model):
                return LAYERNORM_SCHEMA

        Example (dynamic schema):
            @classmethod
            def build_schema(cls, node, model):
                num_inputs = len(node.input)
                inputs = [InputSchema(name=f"input{i}", ...) for i in range(num_inputs)]
                return KernelSchema(name="Concat", inputs=inputs, outputs=[...])
        """
        raise NotImplementedError(f"{cls.__name__}.build_schema()")

    # ====================================================================
    # Inference Support (Unified System)
    # ====================================================================

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Check if this kernel can transform the given ONNX node (default: no)."""
        return False

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int
    ) -> TransformationResult:
        """Transform ONNX node to hardware kernel node(s)."""
        raise NotImplementedError(f"{cls.__name__}.infer_from()")

    def _error(self, message: str) -> KernelOpError:
        """Create exception with node context."""
        return KernelOpError(self.onnx_node, message)

    # ====================================================================
    # Public API: FINN Integration
    # ====================================================================

    def get_nodeattr_types(self):
        """Return nodeattr registry (datatypes + user params + kernel params).

        Auto-delegates to kernel_schema.build_nodeattr_registry() which includes:
        - Interface datatypes (input0Datatype, output0Datatype, etc.)
        - Internal datatypes (accumulatorDatatype, etc.)
        - Template parameters (SIMD, PE, etc.)
        - Kernel-specific parameters (epsilon, algorithm, etc.)

        Automatically sets FIFO depth defaults based on kernel schema interface counts.

        Only override if build_schema() needs to read nodeattrs (circular dependency).
        In that case, define nodeattrs explicitly before calling build_schema().
        """
        base_attrs = super().get_nodeattr_types()

        # Add implementation attribute for backend selection
        # Replaces domain mutation used by FINN's SpecializeLayers
        base_attrs.update(
            {
                "implementation": (
                    "s",
                    False,
                    "",
                    {
                        "",  # Not yet specialized
                        "vitis_hls",  # Vitis HLS (2020.1+)
                        "verilog",  # Verilog RTL
                        "systemverilog",  # SystemVerilog RTL
                        "static_ip",  # Pre-generated IP core
                    },
                ),
            }
        )

        try:
            base_attrs.update(self.kernel_schema.build_nodeattr_registry())
        except RecursionError as e:
            raise RuntimeError(
                f"{self.__class__.__name__}.kernel_schema property calls get_nodeattr(), "
                f"creating circular dependency. You must override get_nodeattr_types() "
                f"explicitly to define nodeattrs before schema construction. "
                f"See KernelOp docstring for mode-dependent schema pattern."
            ) from e

        # Auto-configure FIFO depths based on schema interface counts
        # This prevents IndexError in FINN's InsertFIFO transform for multi-input/output kernels
        num_inputs = len(self.kernel_schema.inputs)
        num_outputs = len(self.kernel_schema.outputs)

        if num_inputs > 0:
            base_attrs["inFIFODepths"] = ("ints", False, [2] * num_inputs)
        if num_outputs > 0:
            base_attrs["outFIFODepths"] = ("ints", False, [2] * num_outputs)

        return base_attrs

    # ====================================================================
    # Public API: Cached State Access (Properties)
    # ====================================================================

    @property
    def design_space(self) -> KernelDesignSpace:
        """Cached design space (call method with model_w first to initialize)."""
        if self._design_space is None:
            raise RuntimeError(
                f"{self.onnx_node.name}: Not initialized. "
                f"Call a method with model_w parameter first."
            )
        return self._design_space

    @property
    def design_point(self) -> KernelDesignPoint:
        """Current kernel configuration as design point (regenerated from nodeattrs).

        This property regenerates on every access to ensure consistency with current
        nodeattrs. For better performance when accessing multiple properties, cache
        the design point in a local variable:

        Example:
            # GOOD: Cache locally for multiple accesses
            point = self.design_point
            simd = point.inputs["input"].stream_shape[-1]
            width = point.inputs["input"].tensor_shape[-1]
            dtype = point.inputs["input"].datatype

            # AVOID: Multiple accesses trigger multiple rebuilds
            simd = self.design_point.inputs["input"].stream_shape[-1]
            width = self.design_point.inputs["input"].tensor_shape[-1]
            dtype = self.design_point.inputs["input"].datatype
        """
        if self._design_space is None:
            raise RuntimeError(
                f"{self.onnx_node.name}: Design space not initialized. "
                f"Call a method with model_w parameter first."
            )

        # Always regenerate design point from current nodeattrs
        current_config = {
            param_name: self.get_nodeattr(param_name)
            for param_name in self._design_space.parameters.keys()
        }

        try:
            return self._design_space.configure(current_config)
        except ValueError as e:
            raise self._error(str(e))

    # ====================================================================
    # Public API: Cache Management
    # ====================================================================

    def invalidate(self) -> None:
        """Invalidate cached design space after external graph changes.

        Call this after transforms that change:
        - Tensor shapes (padding, reshape)
        - Datatypes in graph metadata
        - Node rewiring (FIFO insertion)

        Next method call with model_w will rebuild design space automatically.
        Design points regenerate on every access, so no explicit invalidation needed.

        Example:
            >>> # After transform changes graph
            >>> model = ApplyPadding().apply(model)
            >>> for node in model.graph.node:
            ...     op = getCustomOp(node)
            ...     if isinstance(op, KernelOp):
            ...         op.invalidate()
        """
        self._design_space = None

    # ====================================================================
    # Private API: Lazy Initialization
    # ====================================================================

    def _ensure_ready(self, model_w: ModelWrapper) -> None:
        """Ensure kernel is ready: initialize design space.

        Idempotent operation that builds design space if not cached.
        Design points are regenerated on-demand from nodeattrs (not cached).

        Args:
            model_w: ModelWrapper for ONNX graph access

        Raises:
            KernelOpError: If construction or validation fails
        """
        if model_w is None:
            raise self._error("ModelWrapper required")

        # Build design space if not cached (expensive, once per structure)
        if self._design_space is None:
            build_ctx = BuildContext(
                schema=self.kernel_schema,
                model_w=model_w,
                node=self.onnx_node,
                param_getter=self.get_nodeattr,
                param_setter=self.set_nodeattr,
            )

            try:
                self._design_space = DesignSpaceBuilder().build(build_ctx)
                logger.debug(f"{self.onnx_node.name}: Built design space")
            except ValueError as e:
                raise self._error(str(e))

            # Auto-populate missing parameter values from auto-computed defaults
            from qonnx.util.basic import get_by_name

            for param_name, param in self._design_space.parameters.items():
                # Check if attribute is actually set in ONNX node (not just has a default)
                if get_by_name(self.onnx_node.attribute, param_name) is None:
                    # Parameter not set - use default/minimum value
                    if isinstance(param, OrderedParameter):
                        # OrderedParameter: use get_default() (explicit default or minimum)
                        initial_value = param.get_default()
                    else:  # frozenset
                        # Defensive: skip empty parameter sets (shouldn't happen with new design)
                        if len(param) == 0:
                            logger.debug(
                                f"{self.onnx_node.name}: Skipping empty parameter {param_name}"
                            )
                            continue
                        # Discrete: use sorted first value
                        initial_value = sorted(param)[0]

                    self.set_nodeattr(param_name, initial_value)
                    logger.debug(
                        f"{self.onnx_node.name}: Auto-populated {param_name}={initial_value}"
                    )

    def _ensure_initialized_for_execution(self, graph):
        """Ensure design_space initialized from graph context (for QONNX executor).

        QONNX executor creates fresh instances per node execution, losing cached
        design_space. This method reconstructs ModelWrapper from the GraphProto
        to initialize lazy state before execution.

        This is a defensive guard for QONNX execution compatibility. When KernelOp
        instances are created via getCustomOp(node) during execute_onnx(), they
        lack the model context needed for lazy initialization. By reconstructing
        ModelWrapper from the graph parameter, we can safely initialize on demand.

        Args:
            graph: ONNX GraphProto (from execute_node parameter)

        Note:
            - Called at start of execute_node() if _design_space is None
            - Idempotent: safe to call multiple times (checks cache first)
            - Minimal overhead: only rebuilds if needed (~1ms for small graphs)
            - Required for Python execution via QONNX executor
            - Not needed for cppsim/rtlsim (use prepared instances directly)
            - Design points regenerate automatically, no caching needed

        Example:
            def execute_node(self, context, graph):
                # Ensure initialized (QONNX executor creates fresh instances)
                self._ensure_initialized_for_execution(graph)

                # Now safe to access design_point (regenerates from design_space)
                dtype = self.design_point.inputs["input"].datatype
                # ... rest of execution
        """
        if self._design_space is None:
            from qonnx.core.modelwrapper import ModelWrapper
            from qonnx.util.basic import qonnx_make_model

            # Reconstruct ModelWrapper from GraphProto
            # This provides tensor shapes/datatypes needed for design space
            model_proto = qonnx_make_model(graph)
            model_w = ModelWrapper(model_proto)

            # Initialize design space (design point regenerates on access)
            self._ensure_ready(model_w)

    def build_design_space(self, model_w: ModelWrapper) -> None:
        """FINN API compatibility: Build design space.

        This method provides compatibility with FINN's getHWCustomOp() utility,
        which detects KernelOp via kernel_schema attribute and calls this method.

        Args:
            model_w: ModelWrapper for graph context
        """
        # Delegate to existing lazy initialization
        self._ensure_ready(model_w)

    def get_valid_ranges(
        self, model_w: ModelWrapper
    ) -> dict[str, Union["OrderedParameter", frozenset]]:
        """Valid parameter values for DSE (tiling + resource).

        Returns:
            Dict mapping parameter names to OrderedParameter (ordered sequences)
            or frozenset (discrete categories).
        """
        self._ensure_ready(model_w)
        return self.design_space.parameters

    def infer_node_datatype(self, model_w):
        """Sync datatypes: model → nodeattrs (inputs), nodeattrs → model (outputs).

        Initializes design space which syncs input datatypes from model to nodeattrs.
        Then propagates output datatypes from nodeattrs back to model.
        """
        # Initialize (syncs inputs: model → nodeattrs)
        self._ensure_ready(model_w)

        # Propagate output datatypes: nodeattrs → model
        for i, out_name in enumerate(self.onnx_node.output):
            if out_name:
                odt = self.get_output_datatype(i)
                model_w.set_tensor_datatype(out_name, odt)

    # ====================================================================
    # Public API: Shape/Datatype Queries
    # ====================================================================

    def get_input_datatype(self, ind=0) -> DataType:
        """Get input datatype."""
        return DataType[self.get_nodeattr(f"input{ind}Datatype")]

    def get_output_datatype(self, ind=0) -> DataType:
        """Get output datatype."""
        return DataType[self.get_nodeattr(f"output{ind}Datatype")]

    # FINN API compatibility - delegate to design_point
    def get_normal_input_shape(self, ind=0) -> tuple[int, ...]:
        """Return normal (unfolded) input shape as immutable tuple (FINN convention)."""
        return self.design_point.input_list[ind].tensor_shape

    def get_normal_output_shape(self, ind=0) -> tuple[int, ...]:
        """Return normal (unfolded) output shape as immutable tuple (FINN convention)."""
        return self.design_point.output_list[ind].tensor_shape

    def get_folded_input_shape(self, ind=0) -> tuple[int, ...]:
        tensor_shape = self.design_point.input_list[ind].tensor_shape
        stream_shape = self.design_point.input_list[ind].stream_shape
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
        flattened_stream = math.prod(stream_shape)
        return tuple(fold_factors + [flattened_stream])

    def get_folded_output_shape(self, ind=0) -> tuple[int, ...]:
        tensor_shape = self.design_point.output_list[ind].tensor_shape
        stream_shape = self.design_point.output_list[ind].stream_shape
        fold_factors = [t // s for t, s in zip(tensor_shape, stream_shape)]
        flattened_stream = math.prod(stream_shape)
        return tuple(fold_factors + [flattened_stream])

    def get_instream_width(self, ind=0) -> int:
        return self.design_point.input_list[ind].stream_width_bits

    def get_outstream_width(self, ind=0) -> int:
        return self.design_point.output_list[ind].stream_width_bits

    def get_number_output_values(self):
        """Get iteration count(s) for output values.

        Matches FINN API pattern:
        - Single-output kernels: Returns int (iteration count)
        - Multi-output kernels: Returns dict mapping output names → iteration counts

        Returns:
            int: For single-output kernels (e.g., MVAU, Thresholding, AddStreams)
            dict: For multi-output kernels (e.g., DuplicateStreams, Split)

        Examples:
            Single-output: 512
            Multi-output: {'out0': 512, 'out1': 512}
        """
        num_outputs = len(self.onnx_node.output)

        if num_outputs == 1:
            # Single-output: Return int (FINN pattern for MVAU, Thresholding, etc.)
            folded_shape = self.get_folded_output_shape(ind=0)
            return math.prod(folded_shape[:-1])
        else:
            # Multi-output: Return dict (FINN pattern for DuplicateStreams, Split)
            out_val = {}
            for i in range(num_outputs):
                folded_shape = self.get_folded_output_shape(ind=i)
                iteration_count = math.prod(folded_shape[:-1])
                out_val[f"out{i}"] = iteration_count
            return out_val

    def get_exp_cycles(self) -> int:
        return self.design_point.initiation_interval

    def make_shape_compatible_op(self, model_w):
        """Create standard ONNX op for shape inference (auto-detects pattern)."""
        from onnx import helper

        num_out = len(self.onnx_node.output)
        num_in = len(self.onnx_node.input)

        if num_in == 1 and num_out > 1:
            return helper.make_node(
                "Split",
                inputs=[self.onnx_node.input[0]],
                outputs=list(self.onnx_node.output),
                axis=-1,
            )

        if num_out == 1:
            input_shapes = [tuple(model_w.get_tensor_shape(inp)) for inp in self.onnx_node.input]

            if len(set(input_shapes)) == 1:
                return super().make_const_shape_op(input_shapes[0])
            else:
                raise NotImplementedError(
                    f"{self.__class__.__name__}: {num_in} inputs with different shapes "
                    f"{input_shapes}. Override make_shape_compatible_op()."
                )

        raise NotImplementedError(
            f"{self.__class__.__name__}: {num_in} inputs → {num_out} outputs. "
            f"Override make_shape_compatible_op()."
        )

    def set_nodeattr(self, name: str, value: Any) -> None:
        """Set nodeattr and auto-invalidate design space if needed.

        Design points regenerate on each access, so no explicit invalidation needed.
        """
        try:
            old_value = self.get_nodeattr(name)
        except (AttributeError, Exception):
            old_value = None

        if old_value != value:
            super().set_nodeattr(name, value)

            # Only invalidate design space for structural changes (datatypes)
            # Dimension changes (PE, SIMD, etc.) don't require invalidation since
            # design points regenerate from nodeattrs on each access
            if name in self.kernel_schema.get_structural_nodeattrs():
                self._design_space = None
            # else: Runtime attributes and dimension changes don't invalidate cache

    def apply_design_point(self, point: "KernelDesignPoint") -> None:
        """Apply chosen design point to nodeattrs (persist to ONNX).

        Syncs design point configuration back to node attributes.
        The design_point property will regenerate from these nodeattrs on next access.

        Args:
            point: Design point to apply

        Raises:
            ValueError: If point from different design space
            RuntimeError: If design space not initialized

        Example:
            >>> # DSE exploration
            >>> best_point = None
            >>> best_cycles = float('inf')
            >>> for point in op.design_space.sweep_dimension("SIMD"):
            ...     if point.initiation_interval < best_cycles:
            ...         best_cycles = point.initiation_interval
            ...         best_point = point
            >>>
            >>> # Apply winner to node
            >>> op.apply_design_point(best_point)
        """
        if self._design_space is None:
            raise RuntimeError(
                f"{self.onnx_node.name}: Design space not initialized. "
                f"Call a method with model_w parameter first."
            )

        if point.design_space is not self._design_space:
            raise ValueError(
                f"{self.onnx_node.name}: DesignPoint from different DesignSpace. "
                f"Cannot apply to this node."
            )

        # Sync config → nodeattrs (bypass set_nodeattr to avoid invalidation)
        for dim_name, value in point.config.items():
            super().set_nodeattr(dim_name, value)

        # Design point will regenerate from nodeattrs on next access
