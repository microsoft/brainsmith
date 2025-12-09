############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Transform to refresh kernel instance cache in KernelOp nodes.

This transform serves two purposes:
1. Refreshes cached kernel instances (design space + configuration) when shapes/types change
2. Can replace InferShapes and InferDataTypes for Brainsmith nodes
"""


from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from brainsmith.dataflow import KernelOp


class RefreshKernelDesignPoints(Transformation):
    """Refresh kernel instance cache for all KernelOp nodes.

    This transform should be called:
    - After any transform that changes tensor shapes
    - After any transform that changes datatypes
    - Before using any KernelOp methods that rely on kernel instances
    - As a replacement for InferShapes/InferDataTypes for Brainsmith ops
    """

    def __init__(self, node_types: list[str] | None = None):
        """Initialize transform.

        Args:
            node_types: Optional list of specific node types to refresh.
                       If None, refreshes all KernelOp nodes.
        """
        super().__init__()
        self.node_types = node_types

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transform to refresh kernel models.

        Returns:
            Tuple of (model, modified) where modified indicates if any
            nodes were refreshed.
        """
        graph_modified = False

        for node in model.graph.node:
            # Skip if node type filtering is enabled and doesn't match
            if self.node_types is not None and node.op_type not in self.node_types:
                continue

            # Try to get custom op instance
            try:
                inst = getCustomOp(node, model=model.model)
            except Exception:
                # Not a custom op or failed to instantiate
                continue

            # Check if it's a KernelOp
            if isinstance(inst, KernelOp):
                # Invalidate and rebuild kernel instance
                inst.invalidate()
                inst.infer_node_datatype(model)  # Initializes and validates
                graph_modified = True

                # Optionally update output shapes/types from kernel instance
                # This makes it a replacement for InferShapes/InferDataTypes
                self._update_tensor_info_from_kernel(model, node, inst)

        return (model, graph_modified)

    def _update_tensor_info_from_kernel(self, model: ModelWrapper, node, op: KernelOp) -> None:
        """Update tensor shapes and datatypes from kernel instance.

        This allows RefreshKernelDesignPoints to serve as a replacement for
        InferShapes and InferDataTypes for Brainsmith operators.
        """
        try:
            design_point = op.design_point  # Returns KernelDesignPoint

            # Update input tensor info
            for i, inp in enumerate(design_point.inputs):
                if i < len(node.input):
                    tensor_name = node.input[i]
                    if tensor_name:  # Skip empty inputs
                        # Update shape
                        model.set_tensor_shape(tensor_name, list(inp.tensor_dims))
                        # Update datatype
                        model.set_tensor_datatype(tensor_name, inp.datatype)

            # Update output tensor info
            for i, out in enumerate(design_point.outputs):
                if i < len(node.output):
                    tensor_name = node.output[i]
                    # Update shape
                    model.set_tensor_shape(tensor_name, list(out.tensor_dims))
                    # Update datatype
                    model.set_tensor_datatype(tensor_name, out.datatype)

        except Exception:
            # Log but don't fail - some ops may not have full info yet
            pass


class InferBrainsmithTypes(Transformation):
    """Infer datatypes for Brainsmith operators using kernel instances.

    This is a more targeted version of RefreshKernelDesignPoints that only
    updates datatypes, similar to qonnx.transformation.infer_datatypes.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply datatype inference."""
        # First refresh all kernel instances
        model, _ = RefreshKernelDesignPoints().apply(model)

        graph_modified = False

        for node in model.graph.node:
            try:
                inst = getCustomOp(node, model=model.model)
                if isinstance(inst, KernelOp):
                    design_point = inst.design_point  # Returns KernelDesignPoint

                    # Update output datatypes only
                    for i, out in enumerate(design_point.outputs):
                        if i < len(node.output):
                            old_dtype = model.get_tensor_datatype(node.output[i])
                            new_dtype = out.datatype
                            if old_dtype != new_dtype:
                                model.set_tensor_datatype(node.output[i], new_dtype)
                                graph_modified = True

            except Exception:
                continue

        return (model, graph_modified)


class InferBrainsmithShapes(Transformation):
    """Infer shapes for Brainsmith operators using kernel instances.

    This is a more targeted version of RefreshKernelDesignPoints that only
    updates shapes, similar to qonnx.transformation.infer_shapes.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply shape inference."""
        # First refresh all kernel instances
        model, _ = RefreshKernelDesignPoints().apply(model)

        graph_modified = False

        for node in model.graph.node:
            try:
                inst = getCustomOp(node, model=model.model)
                if isinstance(inst, KernelOp):
                    design_point = inst.design_point  # Returns KernelDesignPoint

                    # Update output shapes only
                    for i, out in enumerate(design_point.outputs):
                        if i < len(node.output):
                            old_shape = model.get_tensor_shape(node.output[i])
                            new_shape = list(out.tensor_dims)
                            if old_shape != new_shape:
                                model.set_tensor_shape(node.output[i], new_shape)
                                graph_modified = True

            except Exception:
                continue

        return (model, graph_modified)


# Convenience function to create standard cleanup pipeline
def make_brainsmith_cleanup_pipeline():
    """Create a standard cleanup pipeline for Brainsmith models.

    Returns a list of transforms that:
    1. Applies any pending config changes
    2. Refreshes all kernel instances
    3. Cleans up the graph
    """
    from finn.transformation.general import ApplyConfig
    from qonnx.transformation.fold_constants import FoldConstants
    from qonnx.transformation.general import RemoveStaticGraphInputs, RemoveUnusedTensors

    return [
        ApplyConfig(),  # Apply any pending config changes
        RefreshKernelDesignPoints(),  # Refresh all kernel instances
        FoldConstants(),  # Fold any constants
        RemoveUnusedTensors(),  # Clean up unused tensors
        RemoveStaticGraphInputs(),  # Remove static inputs
    ]
