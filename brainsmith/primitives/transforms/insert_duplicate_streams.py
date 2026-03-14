############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Insert DuplicateStreams layers for tensor fanout."""


import logging

from onnx import TensorProto, helper
from onnx.onnx_pb import StringStringEntryProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

logger = logging.getLogger(__name__)


class InsertDuplicateStreams(Transformation):
    """Insert DuplicateStreams HW layer for any tensor with fanout >= 2.

    Unlike node-pattern inference (Add→ChannelwiseOp), this is a graph-level
    transform that detects multi-consumer tensors and inserts routing layers.

    Usage:
        model = model.transform(InsertDuplicateStreams())

    Example:
        Before:
            Conv → tensor_X → [Add_1, Add_2, Mul_3]

        After:
            Conv → tensor_X → DuplicateStreams → [clone_0, clone_1, clone_2]
                                                    ↓        ↓        ↓
                                                  Add_1   Add_2    Mul_3
    """

    def __init__(self):
        super().__init__()

    def apply(self, model: ModelWrapper):
        """Scan graph and insert DuplicateStreams for multi-consumer tensors."""
        graph = model.graph
        graph_modified = False
        node_ind = 0

        # Check if global input needs duplication
        if graph.input:
            global_input = graph.input[0].name
            if self._needs_duplication(model, global_input):
                self._insert_duplicator(model, global_input, 0)
                graph_modified = True
                node_ind += 1

        # Check all node outputs
        for node in list(graph.node):  # List copy - we'll modify graph
            for output_tensor in node.output:
                if self._needs_duplication(model, output_tensor):
                    self._insert_duplicator(model, output_tensor, node_ind)
                    graph_modified = True
                    node_ind += 1
            node_ind += 1

        # Cleanup after modifications
        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)

    def _needs_duplication(self, model: ModelWrapper, tensor_name: str) -> bool:
        """Check if tensor has fanout >= 2.

        Args:
            model: ONNX model wrapper
            tensor_name: Tensor to check

        Returns:
            True if tensor feeds 2+ consumers
        """
        successors = model.find_consumers(tensor_name)
        return successors is not None and len(successors) >= 2

    def _insert_duplicator(
        self, model: ModelWrapper, output_tensor: str, insert_index: int
    ) -> None:
        """Insert DuplicateStreams node and rewire consumers.

        Args:
            model: ONNX model wrapper
            output_tensor: Tensor with multiple consumers
            insert_index: Where to insert new node
        """
        graph = model.graph
        successors = model.find_consumers(output_tensor)
        n_outputs = len(successors)

        # Get tensor metadata from graph
        out_shape = model.get_tensor_shape(output_tensor)
        dt = model.get_tensor_datatype(output_tensor)

        # Create clone tensors (one per consumer)
        out_tensor_clones = []
        for i in range(n_outputs):
            clone = helper.make_tensor_value_info(
                model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
            )
            graph.value_info.append(clone)
            model.set_tensor_datatype(clone.name, dt)  # Preserve datatype
            out_tensor_clones.append(clone.name)

        # Create DuplicateStreams node
        # Modern: No shape nodeattrs, minimal attributes
        dup_node = helper.make_node(
            "DuplicateStreams",
            inputs=[output_tensor],
            outputs=out_tensor_clones,
            name=f"DuplicateStreams_{output_tensor.replace('/', '_')}",
            domain="brainsmith.kernels",
        )

        # Set backend attribute to enable specialization
        # Required by FINN's SpecializeKernel transform (line 68-76)
        dup_node.attribute.append(helper.make_attribute("backend", "fpgadataflow"))

        # Copy PyTorch hierarchy metadata for MLO loop rolling
        # Infrastructure kernels must inherit hierarchy from consumers (they exist to serve them)
        metadata_copied = self._copy_hierarchy_metadata(
            dup_node, successors, model, output_tensor
        )

        if not metadata_copied:
            logger.debug(
                f"DuplicateStreams for {output_tensor}: no hierarchy metadata found "
                f"(may be excluded from FINNLoop)"
            )

        # Insert node into graph
        graph.node.insert(insert_index, dup_node)

        # Rewire consumers to use clone tensors
        clone_idx = 0
        for successor in successors:
            for i, succ_input in enumerate(successor.input):
                if succ_input == output_tensor:
                    successor.input[i] = out_tensor_clones[clone_idx]
                    clone_idx += 1
                    # Break inner loop - one clone per consumer connection
                    break

    def _copy_hierarchy_metadata(
        self,
        dup_node,
        successors,
        model: ModelWrapper,
        output_tensor: str
    ) -> bool:
        """Copy PyTorch hierarchy metadata from consumers to DuplicateStreams node.

        For MLO loop rolling, nodes need pkg.torch.onnx.name_scopes and
        pkg.torch.onnx.class_hierarchy metadata to be included in FINNLoop bodies.

        Infrastructure kernels inherit from consumers (not producers) because:
        - Consumers define where the duplicated data is needed
        - Validates all consumers in same hierarchy (no cross-loop fanout)
        - More robust than producer (which may be optimized away)

        Args:
            dup_node: DuplicateStreams ONNX node to annotate
            successors: Consumer nodes
            model: ModelWrapper
            output_tensor: Tensor being duplicated

        Returns:
            True if metadata was copied, False otherwise
        """
        METADATA_KEYS = ["pkg.torch.onnx.name_scopes", "pkg.torch.onnx.class_hierarchy"]

        # Collect metadata from all consumers
        consumer_metadata = []
        for consumer in successors:
            consumer_meta = {}
            for prop in consumer.metadata_props:
                if prop.key in METADATA_KEYS:
                    consumer_meta[prop.key] = prop.value
            if consumer_meta:
                consumer_metadata.append(consumer_meta)

        # No metadata found in any consumer
        if not consumer_metadata:
            # Fall back to producer
            producer = model.find_producer(output_tensor)
            if producer:
                for prop in producer.metadata_props:
                    if prop.key in METADATA_KEYS:
                        # Use StringStringEntryProto for metadata_props
                        new_prop = StringStringEntryProto(key=prop.key, value=prop.value)
                        dup_node.metadata_props.append(new_prop)
                return len([p for p in producer.metadata_props if p.key in METADATA_KEYS]) > 0
            return False

        # For loop rolling, what matters is the common prefix, not exact match
        # E.g., "encoder.layer.0.attention.self.query" and "encoder.layer.0.attention.self.key"
        # both belong to the same loop iteration (encoder.layer.0)

        # Find longest common prefix for name_scopes
        name_scopes_list = []
        for meta in consumer_metadata:
            scope_str = meta.get("pkg.torch.onnx.name_scopes", "")
            # Parse as list (format: ['encoder', 'encoder.layer.0', ...])
            try:
                import ast
                scope_list = ast.literal_eval(scope_str)
                name_scopes_list.append(scope_list)
            except:
                # If parsing fails, treat as incompatible
                name_scopes_list.append([])

        # Find common prefix across all consumers
        if name_scopes_list and all(name_scopes_list):
            common_prefix = name_scopes_list[0]
            for scopes in name_scopes_list[1:]:
                # Find longest common prefix
                common_prefix = [
                    common_prefix[i]
                    for i in range(min(len(common_prefix), len(scopes)))
                    if i < len(scopes) and common_prefix[i] == scopes[i]
                ]

            # Use common prefix as the hierarchy for DuplicateStreams
            if common_prefix:
                # Reconstruct metadata using common prefix
                common_hierarchy_str = str(common_prefix)

                # Get class hierarchy from first consumer (should be same at prefix level)
                class_hierarchy = consumer_metadata[0].get("pkg.torch.onnx.class_hierarchy", "")

                new_prop = StringStringEntryProto(
                    key="pkg.torch.onnx.name_scopes",
                    value=common_hierarchy_str
                )
                dup_node.metadata_props.append(new_prop)

                if class_hierarchy:
                    new_prop = StringStringEntryProto(
                        key="pkg.torch.onnx.class_hierarchy",
                        value=class_hierarchy
                    )
                    dup_node.metadata_props.append(new_prop)

                logger.debug(
                    f"DuplicateStreams for {output_tensor}: using common prefix {common_prefix}"
                )
                return True

        # Fallback: use first consumer's full metadata
        reference_metadata = consumer_metadata[0]
        for key, value in reference_metadata.items():
            new_prop = StringStringEntryProto(key=key, value=value)
            dup_node.metadata_props.append(new_prop)

        return True
