############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Generic single-kernel inference transform.

This module provides InferKernel, a modular transform for inferring a single
hardware kernel from ONNX nodes. It delegates to the kernel's declarative
inference methods (can_infer_from, infer_from).

Example usage:
    from brainsmith.primitives.transforms.infer_kernel import InferKernel
    from brainsmith.kernels.addstreams import AddStreams

    # Infer a specific kernel
    model = model.transform(InferKernel(AddStreams))
"""

import logging

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

logger = logging.getLogger(__name__)


class InferKernel(Transformation):
    """Transform to infer a specific kernel from ONNX nodes.

    This transform provides modular, single-kernel inference by delegating
    to the kernel class's inference methods. It's the building block for
    the more general InferKernelList meta-transform.

    The transform iterates through ONNX nodes and:
    1. Checks if the kernel can infer from each node (via can_infer_from)
    2. Performs the inference (via infer_from)
    3. Applies graph modifications (insert/remove nodes)
    4. Re-runs shape/datatype inference if modified

    Args:
        kernel_cls: KernelOp subclass to infer (must implement inference interface)

    Raises:
        ValueError: If kernel_cls is not a KernelOp subclass
        NotImplementedError: If kernel doesn't implement required inference methods

    Example:
        # Infer AddStreams from ONNX Add nodes
        from brainsmith.kernels.addstreams import AddStreams
        model = model.transform(InferKernel(AddStreams))

        # Chain multiple single-kernel transforms
        model = model.transform(InferKernel(AddStreams))
        model = model.transform(InferKernel(Softmax))
        model = model.transform(InferKernel(LayerNorm))

    Implementation Notes:
        - Only processes nodes where can_infer_from() returns True
        - Skips nodes that fail inference (logs warning, continues)
        - Only re-runs InferShapes/InferDataTypes if graph was modified
        - Uses InferenceResult to structure graph modifications
    """

    def __init__(self, kernel_cls: type):
        """Initialize with kernel class.

        Args:
            kernel_cls: KernelOp subclass to infer

        Raises:
            ValueError: If kernel_cls is not a KernelOp subclass
        """
        super().__init__()

        # Import here to avoid circular dependency
        from brainsmith.dataflow import KernelOp

        if not issubclass(kernel_cls, KernelOp):
            raise ValueError(
                f"InferKernel requires a KernelOp subclass, got {kernel_cls.__name__}. "
                f"For legacy HWCustomOp kernels, use InferKernelList with the kernel class directly."
            )

        self.kernel_cls = kernel_cls
        self.kernel_name = kernel_cls.__name__

    def apply(self, model: ModelWrapper):
        """Apply kernel inference to the model.

        Iterates through ONNX nodes and converts matching nodes to
        hardware kernel instances.

        Args:
            model: QONNX ModelWrapper to transform

        Returns:
            Tuple of (transformed_model, graph_modified_flag)
        """
        graph = model.graph
        graph_modified = False

        # Track statistics for logging
        nodes_processed = 0
        nodes_converted = 0
        nodes_failed = 0
        kernel_index = 0  # Track sequential index for this kernel type

        # Iterate nodes (copy list since we'll modify it)
        for node_ind, node in enumerate(list(graph.node)):
            try:
                # Check if this kernel can infer from this node
                # (can_infer_from already validates schema constraints)
                if not self.kernel_cls.can_infer_from(node, model):
                    continue

                nodes_processed += 1
                logger.debug(f"Inferring {self.kernel_name} from {node.op_type} node {node.name}")

                # Delegate to kernel-specific inference (naive node creation)
                # Pass kernel_index for sequential naming
                result = self.kernel_cls.infer_from(node, model, node_ind + 1, kernel_index=kernel_index)

                # VALIDATE new kernel nodes before applying transformation
                # Try to create KernelOp instances and validate design space
                for new_node in result.nodes_to_insert:
                    # Only validate KernelOp nodes (skip layout conversion nodes, etc.)
                    if new_node.domain == "brainsmith.kernels":
                        try:
                            # Attempt to create KernelOp and validate constraints
                            from qonnx.custom_op.registry import getCustomOp

                            kernel_op = getCustomOp(new_node)
                            kernel_op.infer_node_datatype(model)  # Initializes and validates
                        except Exception as e:
                            # Validation failed - skip this transformation
                            logger.debug(
                                f"Skipping {self.kernel_name} inference from {node.name}: "
                                f"validation failed: {e}"
                            )
                            raise  # Re-raise to outer catch block

                # All validations passed - apply graph modifications

                # Ensure opset import exists for new kernel domain
                for new_node in result.nodes_to_insert:
                    if new_node.domain:  # Only add if node has a domain
                        existing_domains = {op.domain for op in model.model.opset_import}
                        if new_node.domain not in existing_domains:
                            import onnx.helper as oh
                            model.model.opset_import.append(oh.make_opsetid(new_node.domain, 1))
                            logger.debug(f"  Added opset import for domain: {new_node.domain}")

                for i, new_node in enumerate(result.nodes_to_insert):
                    graph.node.insert(node_ind + 1 + i, new_node)
                    logger.debug(f"  Inserted {new_node.op_type} node {new_node.name}")

                for old_node in result.nodes_to_remove:
                    graph.node.remove(old_node)
                    logger.debug(f"  Removed {old_node.op_type} node {old_node.name}")

                # Log metadata if present
                if result.metadata:
                    logger.debug(f"  Metadata: {result.metadata}")

                nodes_converted += 1
                kernel_index += 1  # Increment index after successful conversion
                graph_modified = True

            except Exception as e:
                nodes_failed += 1
                logger.warning(
                    f"Failed to infer {self.kernel_name} from {node.op_type} node {node.name}: {e}",
                    exc_info=True,
                )
                # Continue to next node (don't fail entire transform)

        # Log summary
        if nodes_processed > 0:
            logger.debug(
                f"InferKernel({self.kernel_name}): processed {nodes_processed} nodes, "
                f"converted {nodes_converted}, failed {nodes_failed}"
            )

        # Re-run shape/datatype inference if graph was modified
        if graph_modified:
            logger.debug(f"{self.kernel_name}: Running InferShapes and InferDataTypes")
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)
