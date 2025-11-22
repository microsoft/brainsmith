# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Core Dataflow Compilation Steps

Core steps for building and specializing the hardware dataflow graph:
- Dataflow graph construction (infrastructure + computational kernel inference)
- Backend specialization (HLS/RTL selection and dataflow partitioning)

These steps form the central compilation pipeline for dataflow accelerators.
"""

import logging
import os
from typing import Any

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.util.basic import getHWCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    ApplyConfig,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.config import extract_model_config_to_json

from brainsmith.primitives.transforms import InferKernels, InsertInfrastructureKernels
from brainsmith.primitives.transforms.specialize_kernels import SpecializeKernels
from brainsmith.registry import get_component_metadata, get_kernel, step

logger = logging.getLogger(__name__)


# === Dataflow Graph Construction ===


@step(name="build_dataflow_graph")
def build_dataflow_graph(model: Any, cfg: Any) -> Any:
    """Build complete dataflow graph from kernel selections (two-phase workflow).

    Extracts kernel classes from cfg.kernel_selections and splits them into:
    1. Infrastructure kernels (is_infrastructure=True) → InsertInfrastructureKernels
    2. Computational kernels (is_infrastructure=False) → InferKernels

    This two-phase approach ensures infrastructure nodes (DuplicateStreams, FIFO, etc.)
    are inserted first via topology analysis, then computational nodes are pattern-matched.

    Args:
        model: ONNX model to transform
        cfg: Build configuration with kernel_selections attribute

    Returns:
        Transformed model with complete dataflow graph (infrastructure + computational kernels)
    """
    kernel_selections = getattr(cfg, "kernel_selections", None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping inference")
        return model

    logger.debug(f"Processing {len(kernel_selections)} kernel(s)...")

    # Split kernel classes into infrastructure and computational
    infrastructure_kernels = []
    computational_kernels = []

    for kernel_name, _ in kernel_selections:
        try:
            kernel_class = get_kernel(kernel_name)
            metadata = get_component_metadata(kernel_name, "kernel")

            if metadata.is_infrastructure:
                infrastructure_kernels.append(kernel_class)
                logger.debug(f"  {kernel_name} (infrastructure)")
            else:
                computational_kernels.append(kernel_class)
                logger.debug(f"  {kernel_name} (computational)")
        except KeyError:
            logger.error(f"  Kernel not found in registry: {kernel_name}")

    # Phase 1: Insert infrastructure kernels via topology analysis
    if infrastructure_kernels:
        logger.debug(f"Inserting {len(infrastructure_kernels)} infrastructure kernel(s)...")
        model = model.transform(InsertInfrastructureKernels(infrastructure_kernels))

    # Phase 2: Infer computational kernels via pattern matching
    if computational_kernels:
        logger.debug(f"Inferring {len(computational_kernels)} computational kernel(s)...")
        model = model.transform(InferKernels(computational_kernels))

    # Ensure all nodes have unique names after graph construction
    # Some legacy FINN transforms (e.g., InferElementwiseBinaryOperation) create
    # nodes without names, which causes issues in downstream steps like partitioning
    model = model.transform(GiveUniqueNodeNames())
    logger.debug("Assigned unique names to all nodes after dataflow graph construction")

    return model


@step(name='insert_infrastructure_kernels')
def insert_infrastructure_kernels_step(model: Any, cfg: Any) -> Any:
    """Insert infrastructure kernels via topology analysis (Phase 1 of dataflow graph build).

    Infrastructure kernels are inserted based on graph topology and connectivity patterns,
    rather than pattern matching. Examples include:
    - DuplicateStreams (for fan-out)
    - FIFOs (for buffering)
    - AddStreams (for fan-in)

    This step extracts infrastructure kernels from cfg.kernel_selections (those with
    is_infrastructure=True metadata) and applies InsertInfrastructureKernels transform.

    Use this step when you want finer control over the build pipeline, running
    infrastructure insertion separately from computational kernel inference.

    Args:
        model: ONNX model to transform
        cfg: Build configuration with kernel_selections attribute

    Returns:
        Transformed model with infrastructure kernels inserted

    Blueprint usage:
        steps:
          - insert_infrastructure_kernels  # Phase 1: topology-based insertion
          - infer_computational_kernels    # Phase 2: pattern-based inference

    See also:
        - build_dataflow_graph: Combined step that runs both phases
        - infer_computational_kernels: Phase 2 only
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping infrastructure insertion")
        return model

    logger.debug(f"Processing {len(kernel_selections)} kernel selection(s)...")

    # Extract only infrastructure kernels
    infrastructure_kernels = []

    for kernel_name, _ in kernel_selections:
        try:
            kernel_class = get_kernel(kernel_name)
            metadata = get_component_metadata(kernel_name, 'kernel')

            if metadata.is_infrastructure:
                infrastructure_kernels.append(kernel_class)
                logger.debug(f"  {kernel_name} (infrastructure)")
        except KeyError:
            logger.error(f"  Kernel not found in registry: {kernel_name}")

    # Insert infrastructure kernels via topology analysis
    if infrastructure_kernels:
        logger.debug(f"Inserting {len(infrastructure_kernels)} infrastructure kernel(s)...")
        model = model.transform(InsertInfrastructureKernels(infrastructure_kernels))
    else:
        logger.debug("No infrastructure kernels selected, skipping insertion")

    return model


@step(name='infer_computational_kernels')
def infer_computational_kernels_step(model: Any, cfg: Any) -> Any:
    """Infer computational kernels via pattern matching (Phase 2 of dataflow graph build).

    Computational kernels are inferred by matching ONNX node patterns against kernel
    transform patterns. Examples include:
    - MatMul → MVAU
    - LayerNorm → LayerNorm_hls
    - Transpose → Shuffle
    - Add/Mul → ElementwiseBinaryOp

    This step extracts computational kernels from cfg.kernel_selections (those with
    is_infrastructure=False metadata) and applies InferKernels transform.

    Use this step when you want finer control over the build pipeline, running
    computational inference separately from infrastructure insertion.

    Args:
        model: ONNX model to transform
        cfg: Build configuration with kernel_selections attribute

    Returns:
        Transformed model with computational kernels inferred and unique node names

    Blueprint usage:
        steps:
          - insert_infrastructure_kernels  # Phase 1: topology-based insertion
          - infer_computational_kernels    # Phase 2: pattern-based inference

    Implementation notes:
        - Applies GiveUniqueNodeNames after inference to fix legacy FINN transforms
        - Some FINN transforms (e.g., InferElementwiseBinaryOperation) create nodes
          without names, which causes issues in downstream partitioning

    See also:
        - build_dataflow_graph: Combined step that runs both phases
        - insert_infrastructure_kernels: Phase 1 only
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping kernel inference")
        return model

    logger.debug(f"Processing {len(kernel_selections)} kernel selection(s)...")

    # Extract only computational kernels
    computational_kernels = []

    for kernel_name, _ in kernel_selections:
        try:
            kernel_class = get_kernel(kernel_name)
            metadata = get_component_metadata(kernel_name, 'kernel')

            if not metadata.is_infrastructure:
                computational_kernels.append(kernel_class)
                logger.debug(f"  {kernel_name} (computational)")
        except KeyError:
            logger.error(f"  Kernel not found in registry: {kernel_name}")

    # Infer computational kernels via pattern matching
    if computational_kernels:
        logger.debug(f"Inferring {len(computational_kernels)} computational kernel(s)...")
        model = model.transform(InferKernels(computational_kernels))
    else:
        logger.debug("No computational kernels selected, skipping inference")

    # Ensure all nodes have unique names after graph construction
    # Some legacy FINN transforms (e.g., InferElementwiseBinaryOperation) create
    # nodes without names, which causes issues in downstream steps like partitioning
    model = model.transform(GiveUniqueNodeNames())
    logger.debug("Assigned unique names to all nodes after computational kernel inference")

    return model


# === Backend Specialization ===


@step(name='specialize_kernel_backends')
def specialize_kernel_backends(model: Any, cfg: Any) -> Any:
    """Specialize kernel backends via partitioning + backend selection.

    This step combines create_dataflow_partition and specialize_layers into a
    unified transformation that:

    1. **Partitioning Phase**: Separates consecutive groups of HWCustomOp nodes
       into StreamingDataflowPartition nodes, which point to separate ONNX files.
       Only dataflow accelerator synthesis can be performed on these HW subgraphs.

    2. **Specialization Phase**: Converts generic hardware kernel nodes to
       specialized backend implementations (HLS or RTL) based on kernel_selections
       config and constraint checking.

    The step handles both Brainsmith KernelOp nodes and legacy FINN HWCustomOp nodes,
    ensuring compatibility with mixed graphs.

    Args:
        model: ModelWrapper containing the ONNX model with hardware kernel nodes
        cfg: Build configuration with:
            - output_dir: Output directory for intermediate models and configs
            - kernel_selections: Backend priority lists for specialization
            - specialize_layers_config_file: Optional user config for manual overrides

    Returns:
        ModelWrapper containing the specialized dataflow partition model

    Blueprint usage:
        steps:
          - build_dataflow_graph         # Infer kernels first
          - specialize_kernel_backends   # Combined partitioning + specialization
          - apply_folding_config         # Then apply parallelization

    Implementation notes:
        - Creates template_specialize_layers_config.json for user reference
        - Supports single StreamingDataflowPartition only (FINN limitation)
        - Returns the dataflow partition model, not the parent model
        - Saves parent model to intermediate_models/dataflow_parent.onnx if enabled
    """
    logger.debug("Building hardware dataflow graph (partitioning + specialization)...")

    # ========================================================================
    # Phase 1: Create Dataflow Partition
    # ========================================================================

    logger.debug("Phase 1: Creating dataflow partition...")

    partition_dir = os.path.join(cfg.output_dir, "intermediate_models", "supported_op_partitions")

    # Use FINN's CreateDataflowPartition to separate HW nodes
    parent_model = model.transform(CreateDataflowPartition(partition_model_dir=partition_dir))

    # Extract the dataflow partition model
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")

    if len(sdp_nodes) == 0:
        logger.error("No StreamingDataflowPartition nodes found after partitioning")
        logger.error("")
        logger.error("This typically means one or more nodes failed to be converted to hardware:")
        logger.error("  1. Kernel inference failed - ONNX nodes were not matched to any kernel")
        logger.error("     → Check that kernels are listed in blueprint design_space.kernels")
        logger.error("     → Verify nodes are supported by the selected kernels")
        logger.error(
            "  2. Backend specialization failed - kernels lack viable backend implementations"
        )
        logger.error("     → Check that backends are configured in kernel_selections")
        logger.error("     → Verify RTL backend constraints are satisfied (see SpecializeKernels)")
        logger.error("")
        logger.error("Debug steps:")
        logger.error("  - Inspect intermediate_models/ to see which nodes remain")
        logger.error("  - Check logs for kernel inference warnings")
        logger.error("  - Verify all ONNX ops have corresponding kernel transforms")
        raise RuntimeError(
            "No hardware dataflow partition created. "
            "One or more nodes failed kernel inference or backend specialization. "
            "See logs above for details."
        )

    if len(sdp_nodes) > 1:
        logger.warning(
            f"Found {len(sdp_nodes)} StreamingDataflowPartition nodes. "
            "Only single partition is officially supported by FINN."
        )

    # Get the dataflow partition model file
    sdp_node = sdp_nodes[0]
    sdp_node_inst = getHWCustomOp(sdp_node, parent_model)
    dataflow_model_filename = sdp_node_inst.get_nodeattr("model")

    logger.debug(f"Dataflow partition extracted: {dataflow_model_filename}")

    # Save parent model if requested
    if cfg.save_intermediate_models:
        parent_model_path = os.path.join(
            cfg.output_dir, "intermediate_models", "dataflow_parent.onnx"
        )
        parent_model.save(parent_model_path)
        logger.debug(f"Saved parent model: {parent_model_path}")

    # Load the dataflow partition for specialization
    model = ModelWrapper(dataflow_model_filename)

    # Create template config for user reference
    template_config_path = os.path.join(cfg.output_dir, "template_specialize_layers_config.json")
    extract_model_config_to_json(model, template_config_path, ["preferred_impl_style"])
    logger.debug(f"Created template config: {template_config_path}")

    # ========================================================================
    # Phase 2: Specialize Layers
    # ========================================================================

    logger.debug("Phase 2: Specializing hardware layers...")

    # Apply user config if provided (manual overrides)
    if cfg.specialize_layers_config_file is not None:
        logger.debug(f"Applying user config: {cfg.specialize_layers_config_file}")
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.specialize_layers_config_file))

    # Run registry-based backend specialization
    logger.debug("Running registry-based backend specialization...")
    model = model.transform(
        SpecializeKernels(cfg),
        apply_to_subgraphs=True  # Support MLO: specialize kernels in FINNLoop bodies
    )

    # Clean up and infer properties
    logger.debug("Running cleanup transformations...")
    for transform in [
        GiveUniqueNodeNames(),
        InferShapes(),
        InferDataTypes()
    ]:
        model = model.transform(transform, apply_to_subgraphs=True)

    return model


# Backward compatibility alias
@step(name='build_hw_graph')
def build_hw_graph(model: Any, cfg: Any) -> Any:
    """Legacy alias for specialize_kernel_backends (backward compatibility).

    DEPRECATED: Use 'specialize_kernel_backends' instead.

    This alias maintains compatibility with existing blueprints that use
    the old 'build_hw_graph' step name. New blueprints should use the
    clearer 'specialize_kernel_backends' name.

    See specialize_kernel_backends() for full documentation.
    """
    logger.warning(
        "Step 'build_hw_graph' is deprecated. "
        "Use 'specialize_kernel_backends' instead for clarity."
    )
    return specialize_kernel_backends(model, cfg)
