# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith transformation steps.

Model transformation pipeline functions (ONNX/QONNX → dataflow). Loaded during
component discovery to trigger @step decorator registration.

Access via registry:
    from brainsmith import get_step
    step_fn = get_step("qonnx_to_finn_step")
"""

# Topology cleanup steps
from brainsmith.steps.topology_cleanup_steps import (
    finn_topology_cleanup_step,
    import_qonnx_quantization_step,
)

# Topology optimization steps
from brainsmith.steps.topology_optimization_steps import (
    normalize_dataflow_layouts_step,
)

# Core dataflow compilation steps
from brainsmith.steps.core_steps import (
    build_dataflow_graph,
    insert_infrastructure_kernels_step,
    infer_computational_kernels_step,
    specialize_kernel_backends,
    build_hw_graph,  # Deprecated alias
)

# Hardware optimization steps
from brainsmith.steps.hardware_optimization_steps import (
    constrain_folding_and_set_pumped_compute_step,
    apply_parallelization_config_step,
    target_fps_parallelization_step,
    explore_kernel_params_step,
    minimize_bit_width_step,
)

# BERT-specific steps
from brainsmith.steps.bert_steps import (
    bert_topology_cleanup_step,
    bert_cleanup_step,
    bert_streamlining_step,
    shell_metadata_handover_step,
)

__all__ = [
    # Topology cleanup
    'finn_topology_cleanup_step',
    'import_qonnx_quantization_step',
    # Topology optimization
    'normalize_dataflow_layouts_step',
    # Core dataflow compilation
    'build_dataflow_graph',
    'insert_infrastructure_kernels_step',
    'infer_computational_kernels_step',
    'specialize_kernel_backends',
    'build_hw_graph',  # Deprecated
    # Hardware optimization
    'constrain_folding_and_set_pumped_compute_step',
    'apply_parallelization_config_step',
    'target_fps_parallelization_step',
    'explore_kernel_params_step',
    'minimize_bit_width_step',
    # BERT-specific
    'bert_topology_cleanup_step',
    'bert_cleanup_step',
    'bert_streamlining_step',
    'shell_metadata_handover_step',
]
