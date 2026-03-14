# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Topology Optimization Steps

Graph optimization transformations applied after quantization import but before
kernel inference. These optimize the graph topology for dataflow execution.
"""

import logging
from typing import Any

from brainsmith.primitives.transforms.normalize_dataflow_layouts import NormalizeDataflowLayouts
from brainsmith.registry import step

logger = logging.getLogger(__name__)


@step(name="normalize_dataflow_layouts")
def normalize_dataflow_layouts_step(model: Any, cfg: Any) -> Any:
    """
    Normalize all tensor layouts to NHWC (channel-last).

    This preprocessing step converts all NCHW (channel-first) tensors in the graph
    to NHWC (channel-last) layout by inserting Transpose nodes. This ensures that
    all subsequent dataflow kernel operations can assume channel-last layout without
    individual layout checks.

    The transformation preserves the original layout contract for graph outputs by
    inserting reverse Transposes where needed.

    Args:
        model: ONNX model wrapper
        cfg: Build configuration (unused by this step)

    Returns:
        model: Transformed model with normalized layouts

    Usage in blueprint:
        steps:
          - "normalize_dataflow_layouts"  # Add before kernel inference
          - ...
    """
    logger.debug("Normalizing dataflow layouts to NHWC (channel-last)")

    # Apply the transformation (transforms are primitives, use direct import)
    model = model.transform(NormalizeDataflowLayouts())

    logger.debug("Layout normalization complete")

    return model
