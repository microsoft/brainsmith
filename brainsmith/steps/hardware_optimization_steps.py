# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Hardware Optimization Steps

Hardware-specific optimizations including parallelization configuration,
FPS-based auto-parallelization, folding constraints, and parameter exploration.
"""

import logging
from typing import Any

from finn.util.basic import getHWCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from brainsmith.primitives.transforms.parallelization import (
    ApplyParallelizationConfig,
    SetParallelization,
)
from brainsmith.primitives.transforms.set_pumped_compute import SetPumpedCompute
from brainsmith.primitives.transforms.temp_shuffle_fixer import TempShuffleFixer
from brainsmith.registry import step

logger = logging.getLogger(__name__)


@step(name="constrain_folding_and_set_pumped_compute")
def constrain_folding_and_set_pumped_compute_step(model, cfg):
    """Apply optimizations including folding constraints and pumped compute."""
    for transform in [TempShuffleFixer(), SetPumpedCompute()]:
        model = model.transform(transform)
    return model


@step(name="apply_parallelization_config")
def apply_parallelization_config_step(model: Any, cfg: Any) -> Any:
    """Apply parallelization config from JSON file.

    Drop-in replacement for FINN's ApplyConfig for parallelization.
    Works with both FINN HWCustomOp and Brainsmith KernelOp nodes.

    Config file path is read from cfg.folding_config_file (FINN convention).
    """
    config_file = getattr(cfg, "folding_config_file", None)

    if config_file is None:
        logger.warning("No folding_config_file specified in config, skipping parallelization")
        return model

    # Handle FINNLoop node naming before applying config
    model = model.transform(GiveUniqueNodeNames())

    loop_nodes = model.get_nodes_by_op_type("FINNLoop")
    for node in loop_nodes:
        node_inst = getHWCustomOp(node, model)
        loop_body = node_inst.get_nodeattr("body")
        loop_body = loop_body.transform(
            GiveUniqueNodeNames(prefix=node.name + "_")
        )
        node_inst.set_nodeattr("body", loop_body.graph)

    logger.debug(f"Applying parallelization config from: {config_file}")

    # Apply to both top-level and FINNLoop subgraphs
    model = model.transform(
        ApplyParallelizationConfig(config_file),
        apply_to_subgraphs=True
    )

    return model


@step(name="target_fps_parallelization")
def target_fps_parallelization_step(model: Any, cfg: Any) -> Any:
    """Auto-generate parallelization from target FPS.

    Drop-in replacement for FINN's SetFolding/target_fps_parallelization.
    Works with both FINN HWCustomOp and Brainsmith KernelOp nodes.

    Target cycles are calculated from cfg.target_fps and cfg.synth_clk_period_ns:
        target_cycles = (1 / target_fps) / (clock_period_ns * 1e-9)
    """
    target_fps = getattr(cfg, "target_fps", None)

    if target_fps is None:
        logger.warning("No target_fps specified in config, skipping auto-parallelization")
        return model

    # Get clock period (default to 5ns if not specified)
    clock_period_ns = getattr(cfg, "synth_clk_period_ns", 5.0)

    # Calculate target cycles from FPS
    target_cycles = int(1e9 / (target_fps * clock_period_ns))

    logger.debug(
        f"Auto-generating parallelization for target_fps={target_fps}, "
        f"clock={clock_period_ns}ns, target_cycles={target_cycles}"
    )

    # Get optional MVAU weight stream width constraint (default 36 bits)
    mvau_wwidth_max = getattr(cfg, "mvau_wwidth_max", 36)

    # Get optional two-pass relaxation flag (default True)
    two_pass_relaxation = getattr(cfg, "two_pass_relaxation", True)

    # Apply to both top-level and FINNLoop subgraphs
    model = model.transform(
        SetParallelization(
            target_cycles_per_frame=target_cycles,
            mvau_wwidth_max=mvau_wwidth_max,
            two_pass_relaxation=two_pass_relaxation,
        ),
        apply_to_subgraphs=True,
        use_preorder_traversal=False,
    )

    # Post-process FINNLoop bodies to ensure unique names and persist changes
    model = model.transform(GiveUniqueNodeNames())
    loop_nodes = model.get_nodes_by_op_type("FINNLoop")
    for node in loop_nodes:
        node_inst = getHWCustomOp(node, model)
        loop_body = node_inst.get_nodeattr("body")
        loop_body = loop_body.transform(GiveUniqueNodeNames(prefix=node.name + "_"))
        node_inst.set_nodeattr("body", loop_body.graph)

    return model


@step(name="explore_kernel_params")
def explore_kernel_params_step(model, cfg):
    """Parameter exploration for design space exploration (DSE).
    
    Explores different parallelization configurations to find optimal
    hardware resource utilization and performance trade-offs.
    """
    # Import here to avoid circular dependency
    from brainsmith.primitives.transforms.parameter_exploration import ExploreKernelParams
    
    if not hasattr(cfg, 'param_exploration_config'):
        logger.warning("No param_exploration_config specified, skipping parameter exploration")
        return model
    
    logger.debug("Running parameter exploration...")
    model = model.transform(ExploreKernelParams(cfg.param_exploration_config))
    
    return model
