############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Parallelization transformations for mixed FINN + Brainsmith graphs.

This module provides drop-in replacements for FINN's parallelization
transformations that work seamlessly with both legacy FINN nodes and
modern Brainsmith KernelOp nodes.

Key Features:
- Automatic node type detection (FINN vs Brainsmith)
- Unified API for setting parallelization parameters
- JSON config compatibility with FINN
- Support for gradual migration

Architecture:
    FINN nodes → set_nodeattr() directly
    Brainsmith KernelOp → design point API

Parameter Setting Strategy:
    For Brainsmith nodes, parameters are set via design points using this strategy:
    1. Try dimension-based API (most general)
    2. Fall back to interface-based API with heuristics:
       - PE: input[0] or output[0] (channels parallelism)
       - SIMD: input[1] (MVAU weights) or input[0] (SWG data)

    For FINN nodes, parameters are set directly via set_nodeattr().
"""

import inspect
import json
import warnings
from collections.abc import Callable
from typing import Any, Protocol

import finn.custom_op.fpgadataflow.hls.elementwise_binary_hls as elementwise_binary_hls
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.set_folding import divisors
from finn.util.basic import getHWCustomOp
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

# ============================================================================
# Type Hints
# ============================================================================


class KernelOpProtocol(Protocol):
    """Protocol for KernelOp interface (avoiding circular import)."""

    onnx_node: Any
    _design_point: Any
    _design_space: Any

    def build_design_space(self, model: ModelWrapper) -> None:
        ...

    def apply_design_point(self, point: Any) -> None:
        ...

    def get_nodeattr(self, name: str) -> Any:
        ...

    def get_exp_cycles(self) -> int:
        ...

    @property
    def design_point(self) -> Any:
        ...

    @property
    def design_space(self) -> Any:
        ...


NodeInstance = HWCustomOp | KernelOpProtocol


# ============================================================================
# Configuration and Constants
# ============================================================================

# Parameter setting strategies (try in order until one succeeds)
PARAM_STRATEGIES: dict[str, list[str]] = {
    "PE": ["dimension", "input:0", "output:0"],
    "SIMD": ["dimension", "input:1", "input:0"],
}


def _discover_elementwise_binary_ops() -> list[str]:
    """Discover all HLS elementwise binary op types via reflection.

    Returns:
        List of op_type strings for elementwise binary operations
    """
    try:
        base_cls = elementwise_binary_hls.ElementwiseBinaryOperation_hls
        ops = [
            op_type
            for op_type, cls in inspect.getmembers(elementwise_binary_hls, inspect.isclass)
            if issubclass(cls, base_cls) and cls is not base_cls  # Exclude base class
        ]
        return ops
    except (ImportError, AttributeError) as e:
        warnings.warn(f"Failed to discover elementwise binary ops: {e}")
        return []


ELEMENTWISE_BINARY_OPS = _discover_elementwise_binary_ops()

# Ops that use PE parallelism (up to NumChannels)
_PE_OPS = frozenset(
    [
        "AddStreams_hls",
        "ChannelwiseOp_hls",
        "DuplicateStreams_hls",
        "GlobalAccPool_hls",
        "Thresholding_hls",
        "Thresholding_rtl",
        *ELEMENTWISE_BINARY_OPS,
    ]
)

# Ops that use SIMD parallelism (up to NumChannels)
_SIMD_OPS = frozenset(
    [
        "FMPadding_rtl",
        "FMPadding_Pixel_hls",
        "ConvolutionInputGenerator_rtl",
        "QuantSoftmax_hls",
        "Shuffle_hls",
        "LayerNorm_hls",
        "StreamingSplit_hls",
        "StreamingConcat_hls",
    ]
)


# ============================================================================
# Phase 1: Helper Functions
# ============================================================================


def _ensure_design_space(node_inst: NodeInstance, model: ModelWrapper) -> None:
    """Ensure design space is built for node instance.

    Args:
        node_inst: KernelOp instance
        model: ModelWrapper containing the node
    """
    if getattr(node_inst, "_design_space", None) is None:
        node_inst.build_design_space(model)


def _can_set_input_stream(space: Any, idx: int, value: int) -> bool:
    """Check if input stream can accept value without exception.

    Args:
        space: Design space instance
        idx: Stream index
        value: Value to check

    Returns:
        True if stream exists and value is valid
    """
    try:
        if not hasattr(space, "input_streams"):
            return False
        if idx >= len(space.input_streams):
            return False
        # Check if value is in valid range
        stream = space.input_streams[idx]
        if hasattr(stream, "values"):
            return value in stream.values
        return True
    except (AttributeError, IndexError, TypeError):
        return False


def _can_set_output_stream(space: Any, idx: int, value: int) -> bool:
    """Check if output stream can accept value without exception.

    Args:
        space: Design space instance
        idx: Stream index
        value: Value to check

    Returns:
        True if stream exists and value is valid
    """
    try:
        if not hasattr(space, "output_streams"):
            return False
        if idx >= len(space.output_streams):
            return False
        # Check if value is in valid range
        stream = space.output_streams[idx]
        if hasattr(stream, "values"):
            return value in stream.values
        return True
    except (AttributeError, IndexError, TypeError):
        return False


def _try_strategy(
    node_inst: NodeInstance, model: ModelWrapper, param_name: str, value: int, strategy: str
) -> bool:
    """Try single strategy for setting parameter, return True if successful.

    Args:
        node_inst: KernelOp instance
        model: ModelWrapper
        param_name: Parameter name
        value: Value to set
        strategy: Strategy string ("dimension", "input:N", "output:N")

    Returns:
        True if strategy succeeded, False otherwise
    """
    space = node_inst.design_space
    point = node_inst.design_point

    if strategy == "dimension":
        if param_name in space.parameters:
            point = point.with_dimension(param_name, value)
            node_inst.apply_design_point(point)
            return True

    elif strategy.startswith("input:"):
        idx = int(strategy.split(":")[1])
        if _can_set_input_stream(space, idx, value):
            point = point.with_input_stream(idx, value)
            node_inst.apply_design_point(point)
            return True

    elif strategy.startswith("output:"):
        idx = int(strategy.split(":")[1])
        if _can_set_output_stream(space, idx, value):
            point = point.with_output_stream(idx, value)
            node_inst.apply_design_point(point)
            return True

    return False


def get_node_interface(node, model: ModelWrapper) -> tuple[NodeInstance, bool]:
    """Detect node type and return (instance, is_brainsmith) tuple.

    Args:
        node: ONNX node
        model: ModelWrapper containing the node

    Returns:
        Tuple of (node_instance, is_brainsmith)
    """
    from brainsmith.dataflow.kernel_op import KernelOp

    node_inst = getHWCustomOp(node, model)
    is_brainsmith = isinstance(node_inst, KernelOp)

    return node_inst, is_brainsmith


def set_parallelization(
    node_inst: NodeInstance,
    is_brainsmith: bool,
    param_name: str,
    value: int,
    model: ModelWrapper,
    strategies: list[str] | None = None,
) -> None:
    """Set parallelization parameter (PE, SIMD, etc.) on mixed graph nodes.

    Unified API that routes to design point API (Brainsmith) or nodeattr (FINN).

    Args:
        node_inst: HWCustomOp or KernelOp instance
        is_brainsmith: True if Brainsmith KernelOp
        param_name: Parameter name (e.g., "PE", "SIMD")
        value: Parallelization value
        model: ModelWrapper
        strategies: Optional list of strategies to try. If None, uses PARAM_STRATEGIES.

    Raises:
        ValueError: If parameter cannot be set or value is invalid
    """
    if is_brainsmith:
        # Brainsmith: Use design point API with strategy pattern
        _ensure_design_space(node_inst, model)

        # Use provided strategies or default from registry
        strategies = strategies or PARAM_STRATEGIES.get(param_name, ["dimension"])

        # Try each strategy in order until one succeeds
        for strategy in strategies:
            if _try_strategy(node_inst, model, param_name, value, strategy):
                return

        # All strategies failed
        space = node_inst.design_space
        raise ValueError(
            f"Cannot set {param_name}={value} for {node_inst.onnx_node.name}. "
            f"Tried strategies: {strategies}. "
            f"Available parameters: {list(space.parameters.keys())}"
        )

    else:
        # FINN: Direct nodeattr setting
        try:
            node_inst.set_nodeattr(param_name, value)
        except (ValueError, AttributeError, KeyError) as e:
            raise ValueError(
                f"Cannot set {param_name}={value} for {node_inst.onnx_node.name}: {e}"
            ) from e


def get_parallelization(
    node_inst: NodeInstance, is_brainsmith: bool, param_name: str
) -> int | None:
    """Get current parallelization parameter value.

    Args:
        node_inst: HWCustomOp or KernelOp instance
        is_brainsmith: True if Brainsmith KernelOp
        param_name: Parameter name (e.g., "PE", "SIMD")

    Returns:
        Current parameter value, or None if parameter doesn't exist

    Note:
        Returns None for unset parameters, not for errors.
        Use design_space.parameters to check valid parameters.
    """
    if is_brainsmith:
        # Brainsmith: Try design point first (most reliable)
        point = getattr(node_inst, "_design_point", None)
        if point is not None and param_name in point.config:
            return point.config[param_name]

        # Heuristic: infer from interface streams
        if param_name == "PE" and point is not None:
            try:
                return point.get_input_stream_value(0)
            except (IndexError, KeyError, AttributeError):
                pass

        # Final fallback: nodeattrs (legacy)
        return getattr(node_inst, param_name, None)

    else:
        # FINN: Read from nodeattrs
        try:
            return node_inst.get_nodeattr(param_name)
        except (AttributeError, KeyError):
            return None


# ============================================================================
# Phase 2: ApplyParallelizationConfig
# ============================================================================


class ApplyParallelizationConfig(Transformation):
    """Apply parallelization config from JSON file to mixed graphs.

    Brainsmith-compatible version of QONNX's ApplyConfig transformation.
    Works seamlessly with graphs containing both legacy FINN nodes and
    modern Brainsmith KernelOp nodes.

    Key Features:
    - Automatic node type detection
    - Uses design point API for Brainsmith nodes
    - Uses set_nodeattr() for FINN nodes
    - Full JSON config compatibility with FINN
    - Supports defaults by op_type
    - Warning system for missing/unused configs

    JSON Config Format:
        {
            "Defaults": {
                "PE": [8, ["ChannelwiseOp_hls", "AddStreams_hls"]],
                "SIMD": [32, ["MVAU_hls"]]
            },
            "MatMul_0": {"PE": 16, "SIMD": 64},
            "ChannelwiseOp_1": {"PE": 8}
        }

    Args:
        config: Path to JSON file or dict with configuration
        node_filter: Optional filter function to select nodes (default: all nodes)

    Example:
        >>> # Apply config from file
        >>> model = model.transform(
        ...     ApplyParallelizationConfig("folding_config.json")
        ... )
        >>>
        >>> # Apply config from dict
        >>> config = {"MatMul_0": {"PE": 16, "SIMD": 32}}
        >>> model = model.transform(ApplyParallelizationConfig(config))
        >>>
        >>> # Filter to specific nodes
        >>> model = model.transform(
        ...     ApplyParallelizationConfig(
        ...         config,
        ...         node_filter=lambda n: "MVAU" in n.op_type
        ...     )
        ... )

    Migration from FINN:
        Replace:
            from qonnx.transformation.general import ApplyConfig
            model = model.transform(ApplyConfig(config_file))

        With:
            from brainsmith.primitives.transforms import ApplyParallelizationConfig
            model = model.transform(ApplyParallelizationConfig(config_file))
    """

    def __init__(self, config: str | dict[str, Any], node_filter: Callable = lambda x: True):
        super().__init__()
        self.config = config
        self.node_filter = node_filter

    def apply(self, model):
        """Apply configuration to model nodes.

        Args:
            model: ModelWrapper

        Returns:
            Tuple of (model, graph_modified)
                - model: Updated ModelWrapper
                - graph_modified: Always False (single pass sufficient)
        """
        # Load config
        if isinstance(self.config, dict):
            model_config = self.config
        else:
            with open(self.config) as f:
                model_config = json.load(f)

        used_configurations = ["Defaults"] if "Defaults" in model_config else []
        missing_configurations = []

        # Apply configuration to each node
        for node in model.graph.node:
            if not self.node_filter(node):
                continue

            # Get node-specific config
            try:
                node_config = model_config[node.name]
            except KeyError:
                missing_configurations.append(node.name)
                node_config = {}

            # Get node instance
            try:
                node_inst, is_brainsmith = get_node_interface(node, model)
            except Exception:
                # Not a HW node, skip
                continue

            used_configurations.append(node.name)

            # Apply defaults (if specified)
            if "Defaults" in model_config:
                default_values = []
                for key, value in model_config["Defaults"].items():
                    # FINN format: {"PE": [8, ["ChannelwiseOp_hls", "all"]]}
                    if isinstance(value, list) and len(value) >= 2:
                        default_val = value[0]
                        applicable_ops = value[1] if isinstance(value[1], list) else [value[1]]

                        if "all" in applicable_ops or node.op_type in applicable_ops:
                            default_values.append((key, default_val))

                # Apply defaults
                for attr, val in default_values:
                    try:
                        set_parallelization(node_inst, is_brainsmith, attr, val, model)
                    except ValueError as e:
                        warnings.warn(f"Could not apply default {attr}={val} to {node.name}: {e}")

            # Apply node-specific config (overrides defaults)
            for attr, value in node_config.items():
                try:
                    set_parallelization(node_inst, is_brainsmith, attr, value, model)
                except ValueError as e:
                    warnings.warn(f"Could not apply {attr}={value} to {node.name}: {e}")

        # Configuration verification warnings
        if missing_configurations:
            warnings.warn(f"\nNo HW configuration for nodes: {', '.join(missing_configurations)}")

        unused_configs = [x for x in model_config if x not in used_configurations]
        if unused_configs:
            warnings.warn(f"\nUnused HW configurations: {', '.join(unused_configs)}")

        # Single iteration is sufficient
        return (model, False)


# ============================================================================
# Phase 3: SetParallelization - Helper Functions
# ============================================================================


def get_valid_values_for_param(
    node_inst: NodeInstance, is_brainsmith: bool, param_name: str, model: ModelWrapper
) -> list[int]:
    """Get valid values for a parallelization parameter.

    For Brainsmith nodes, extracts valid values from get_valid_ranges().
    For FINN nodes, generates valid values using divisors().

    Args:
        node_inst: HWCustomOp or KernelOp instance
        is_brainsmith: True if Brainsmith KernelOp
        param_name: Parameter name (e.g., "PE", "SIMD")
        model: ModelWrapper

    Returns:
        List of valid values in ascending order
    """
    if is_brainsmith:
        # Brainsmith: Extract from get_valid_ranges()
        from brainsmith.dataflow.ordered_parameter import OrderedParameter

        _ensure_design_space(node_inst, model)
        parameters = node_inst.get_valid_ranges(model)

        if param_name not in parameters:
            return []

        param = parameters[param_name]

        # OrderedParameter has .values tuple
        if isinstance(param, OrderedParameter):
            return list(param.values)
        else:
            # FrozenSet or other iterable
            return sorted(param)

    else:
        # FINN: Generate via divisors(max_val)
        try:
            max_val = get_max_value_for_param(node_inst, param_name)
            return list(divisors(max_val))
        except (AttributeError, ValueError) as e:
            warnings.warn(
                f"Cannot determine valid values for {param_name} on "
                f"{node_inst.onnx_node.name}: {e}"
            )
            return []


def get_max_value_for_param(node_inst: Any, param_name: str) -> int:
    """Get maximum value for a FINN parallelization parameter.

    Maps parameter names (PE, SIMD) to the nodeattr that defines the
    maximum allowed value, which varies by op_type.

    Args:
        node_inst: FINN HWCustomOp instance
        param_name: Parameter name ("PE" or "SIMD")

    Returns:
        Maximum value for the parameter

    Raises:
        ValueError: If parameter/op_type combination is unknown
        AttributeError: If required nodeattr doesn't exist

    Example:
        >>> # MVAU node
        >>> max_pe = get_max_value_for_param(mvau_inst, "PE")
        >>> # Returns: value of MH nodeattr
        >>>
        >>> # ChannelwiseOp node
        >>> max_pe = get_max_value_for_param(channelwise_inst, "PE")
        >>> # Returns: value of NumChannels nodeattr
    """
    op_type = node_inst.onnx_node.op_type

    # PE parameter mappings
    if param_name == "PE":
        if op_type in ["MVAU_hls", "MVAU_rtl"]:
            return node_inst.get_nodeattr("MH")
        elif op_type == "LabelSelect_hls":
            return node_inst.get_nodeattr("Labels")
        elif op_type in ["VVAU_hls", "VVAU_rtl", "Pool_hls"]:
            # Depthwise operations
            return node_inst.get_nodeattr("Channels")
        else:
            # Most PE ops use NumChannels
            # PE ops: AddStreams, ChannelwiseOp, DuplicateStreams,
            #         GlobalAccPool, Thresholding, Elementwise binary ops
            try:
                return node_inst.get_nodeattr("NumChannels")
            except AttributeError:
                # Some recent ops don't have NumChannels, extract from shape
                return node_inst.get_normal_input_shape()[-1]

    # SIMD parameter mappings
    elif param_name == "SIMD":
        if op_type in ["MVAU_hls", "MVAU_rtl"]:
            return node_inst.get_nodeattr("MW")
        elif op_type.startswith("ConvolutionInputGenerator"):
            # Non-depthwise SWG uses IFMChannels
            # (depthwise SWG is handled separately in optimization loop)
            return node_inst.get_nodeattr("IFMChannels")
        elif op_type == "StreamingConcat_hls" or op_type == "StreamingSplit_hls":
            # These use common_divisors, handled specially in optimization loop
            # Return 0 to signal special handling needed
            return 0
        else:
            # Most SIMD ops use NumChannels
            # SIMD ops: FMPadding, QuantSoftmax, Shuffle, LayerNorm
            return node_inst.get_nodeattr("NumChannels")

    else:
        raise ValueError(
            f"Unknown parameter name '{param_name}' for op_type '{op_type}'. "
            f"Expected 'PE' or 'SIMD'."
        )


# ============================================================================
# Phase 3: SetParallelization - Transformation Class
# ============================================================================


class SetParallelization(Transformation):
    """Auto-generate parallelization config for mixed graphs.

    Brainsmith-compatible version of FINN's SetFolding transformation.
    Auto-generates parallelization configuration based on target cycles
    per frame, working seamlessly with both legacy FINN nodes and modern
    Brainsmith KernelOp nodes.

    Key Features:
    - Automatic node type detection
    - Uses get_valid_ranges() for Brainsmith nodes
    - Uses divisors() for FINN nodes
    - MVAU special logic (SIMD first, then PE)
    - Two-pass relaxation support
    - Bottleneck detection and warnings

    Args:
        target_cycles_per_frame: Target performance (cycles per inference)
        mvau_wwidth_max: Max weight stream width for MVAU (default: 36 bits)
        two_pass_relaxation: Enable second pass if target not met (default: True)

    Example:
        >>> # Auto-generate for 60 FPS at 100MHz
        >>> target_fps = 60
        >>> clk_period_ns = 10  # 100MHz
        >>> target_cycles = (1 / target_fps) / (clk_period_ns * 1e-9)
        >>>
        >>> model = model.transform(
        ...     SetParallelization(target_cycles_per_frame=target_cycles)
        ... )

    Migration from FINN:
        Replace:
            from finn.transformation.fpgadataflow.set_folding import SetFolding
            model = model.transform(SetFolding(target_cycles))

        With:
            from brainsmith.primitives.transforms import SetParallelization
            model = model.transform(SetParallelization(target_cycles))
    """

    def __init__(
        self,
        target_cycles_per_frame: int = 1000,
        mvau_wwidth_max: int = 36,
        two_pass_relaxation: bool = True,
    ):
        super().__init__()
        self.target_cycles_per_frame = target_cycles_per_frame
        self.mvau_wwidth_max = mvau_wwidth_max
        self.two_pass_relaxation = two_pass_relaxation

    def optimize_parameter(
        self, node_inst: NodeInstance, is_brainsmith: bool, param_name: str, model: ModelWrapper
    ) -> bool:
        """Optimize single parameter using greedy search.

        Args:
            node_inst: HWCustomOp or KernelOp instance
            is_brainsmith: True if Brainsmith KernelOp
            param_name: Parameter name (e.g., "PE", "SIMD")
            model: ModelWrapper

        Returns:
            True if target cycles met, False if reached max value
        """
        # Get valid values for this parameter
        valid_values = get_valid_values_for_param(node_inst, is_brainsmith, param_name, model)

        if not valid_values:
            return False  # No optimization possible

        # Greedy search: try increasing values until target met
        for val in valid_values:
            set_parallelization(node_inst, is_brainsmith, param_name, val, model)
            cycles = node_inst.get_exp_cycles()

            if cycles < self.target_cycles_per_frame:
                return True  # Target met

        # Reached max value without meeting target
        return False

    def optimize_mvau(self, node_inst: Any, model: Any) -> None:
        """Two-stage optimization for FINN MVAU nodes.

        MVAU nodes require special handling:
        1. Stage 1: Increase SIMD until target met or weight stream too wide
        2. Stage 2: Increase PE until target met or max PE reached

        The weight stream width constraint (default 36 bits) prevents
        memory bandwidth bottlenecks.

        Args:
            node_inst: FINN MVAU HWCustomOp instance
            model: ModelWrapper

        Note:
            This method is FINN-specific and should only be called for
            MVAU_hls or MVAU_rtl nodes.

        Example:
            >>> # MVAU node with MW=64, MH=16
            >>> op_type = node.op_type
            >>> if op_type in ["MVAU_hls", "MVAU_rtl"]:
            ...     self.optimize_mvau(node_inst, model)
        """
        max_simd = node_inst.get_nodeattr("MW")

        # Reset both to minimum
        node_inst.set_nodeattr("PE", 1)
        node_inst.set_nodeattr("SIMD", 1)

        # Stage 1: Increase SIMD (with width constraint)
        wdt = node_inst.get_input_datatype(1)  # Weight datatype
        for simd_val in divisors(max_simd):
            prev_simd = node_inst.get_nodeattr("SIMD")
            node_inst.set_nodeattr("SIMD", simd_val)

            # Check cycle target
            cycles = node_inst.get_exp_cycles()
            if cycles < self.target_cycles_per_frame:
                # Target met
                break

            # Check weight stream width
            weight_stream_width = wdt.bitwidth() * simd_val
            if weight_stream_width > self.mvau_wwidth_max:
                # Revert to previous value (we've gone over width threshold)
                node_inst.set_nodeattr("SIMD", prev_simd)
                break

        # Stage 2: Increase PE
        self.optimize_parameter(node_inst, False, "PE", model)

    def apply(self, model):
        """Apply parallelization optimization.

        Iterates through all HLS/RTL nodes in the graph and optimizes their
        parallelization parameters to meet the target cycles per frame.

        Algorithm:
        1. Categorize nodes by optimization strategy
        2. Apply appropriate optimization method to each node
        3. Annotate cycle estimates
        4. Optionally run second pass if target not achievable (two_pass_relaxation)

        Args:
            model: ModelWrapper

        Returns:
            Tuple of (model, graph_modified)
                - model: Updated ModelWrapper with optimized parallelization
                - graph_modified: Always False (handled via AnnotateCycles)
        """
        from qonnx.transformation.general import GiveUniqueNodeNames

        graph = model.graph

        # Iterate through all nodes and optimize
        for node in graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue

            op_type = node.op_type

            # Get node interface (detects Brainsmith vs FINN)
            try:
                node_inst, is_brainsmith = get_node_interface(node, model)
            except Exception:
                # Not a valid HW node, skip
                continue

            # Dispatch based on node type
            if is_brainsmith:
                # Brainsmith node: Try all parameters from design space
                _ensure_design_space(node_inst, model)

                dimensions = node_inst.get_valid_ranges(model)
                for param_name in dimensions.keys():
                    try:
                        self.optimize_parameter(node_inst, True, param_name, model)
                    except (ValueError, AttributeError) as e:
                        warnings.warn(f"Could not optimize {param_name} for {node.name}: {e}")

            elif op_type in ["MVAU_hls", "MVAU_rtl"]:
                # FINN MVAU: Two-stage optimization (SIMD first, then PE)
                self.optimize_mvau(node_inst, model)

            elif op_type in _PE_OPS:
                # FINN PE ops: Optimize PE parameter
                try:
                    self.optimize_parameter(node_inst, False, "PE", model)
                except (ValueError, AttributeError) as e:
                    warnings.warn(f"Could not optimize PE for {node.name}: {e}")

            elif op_type == "LabelSelect_hls":
                # LabelSelect: Optimize PE (uses Labels attribute)
                try:
                    self.optimize_parameter(node_inst, False, "PE", model)
                except (ValueError, AttributeError) as e:
                    warnings.warn(f"Could not optimize PE for {node.name}: {e}")

            elif op_type in _SIMD_OPS:
                # FINN SIMD ops: Optimize SIMD parameter
                # Skip depthwise ConvolutionInputGenerator (handled by consumer)
                if op_type.startswith("ConvolutionInputGenerator"):
                    depthwise = node_inst.get_nodeattr("depthwise")
                    if depthwise == 1:
                        continue  # Skip depthwise SWG

                try:
                    self.optimize_parameter(node_inst, False, "SIMD", model)
                except (ValueError, AttributeError) as e:
                    warnings.warn(f"Could not optimize SIMD for {node.name}: {e}")

            else:
                # Unknown op_type
                warnings.warn(f"SetParallelization doesn't know how to handle op_type {op_type}")

        # Ensure unique node names and annotate cycles
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())

        # Two-pass relaxation: run again with achievable target if needed
        # Skip if there are unspecialized nodes (e.g., Shuffle) that will be decomposed later
        if self.two_pass_relaxation:
            # Check for abstract/unspecialized nodes that aren't HLS/RTL yet
            has_unspecialized = any(
                node.op_type == "Shuffle" or
                (hasattr(node, 'domain') and node.domain == "finn.custom_op.fpgadataflow.fpgadataflow")
                for node in model.graph.node
            )

            if has_unspecialized:
                warnings.warn(
                    "Skipping two-pass relaxation: model contains unspecialized nodes (e.g., Shuffle) "
                    "that will be decomposed by transpose_decomposition. Parallelization will be "
                    "refined after decomposition if needed."
                )
            else:
                perf_dict = model.analysis(dataflow_performance)
                if perf_dict["max_cycles"] > self.target_cycles_per_frame:
                    # Target not achievable, run second pass with achievable target
                    warnings.warn(
                        f"Node {perf_dict['max_cycles_node_name']} is bottleneck with "
                        f"{perf_dict['max_cycles']} cycles, running second pass"
                    )
                    model = model.transform(
                        SetParallelization(
                            target_cycles_per_frame=perf_dict["max_cycles"],
                            mvau_wwidth_max=self.mvau_wwidth_max,
                            two_pass_relaxation=False,  # Prevent infinite recursion
                        )
                    )

        return (model, False)


__all__ = [
    # Phase 1 Helper functions
    "get_node_interface",
    "set_parallelization",
    "get_parallelization",
    # Phase 3 Helper functions
    "get_valid_values_for_param",
    "get_max_value_for_param",
    # Transformations
    "ApplyParallelizationConfig",
    "SetParallelization",
]
