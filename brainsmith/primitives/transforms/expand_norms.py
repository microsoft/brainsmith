# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Expand Norms Transform

Port of the ExpandNorms transform to the registry system.
"""

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


class ExpandNorms(Transformation):
    """Expand any standard LayerNorms/RMSNorms into the functional
    norm and Mul/Add nodes for affine scale and bias."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        # Collect all LayerNorm nodes and their replacements
        replacements = []

        for node_idx, node in enumerate(graph.node):
            if node.op_type == "LayerNormalization":
                ln_act_in = node.input[0]
                act_out = node.output[0]
                scale = node.input[1]
                bias = node.input[2] if len(node.input) > 2 else None
                axis_attr = get_by_name(node.attribute, "axis")
                axis = axis_attr.i if axis_attr is not None else -1
                epsilon_attr = get_by_name(node.attribute, "epsilon")
                epsilon = epsilon_attr.f if epsilon_attr is not None else 1e-5
                idt = model.get_tensor_datatype(ln_act_in)
                wdt = model.get_tensor_datatype(scale)
                odt = model.get_tensor_datatype(act_out)

                act_shape = model.get_tensor_shape(ln_act_in)

                func_ln_node = oh.make_node(
                    "FuncLayerNorm",
                    [ln_act_in],
                    [act_out],
                    domain="brainsmith.operators.general",
                    backend="general",
                    axis=axis,
                    epsilon=epsilon,
                    InputDataType=idt.name,
                    OutputDataType=odt.name,
                    name=f"FuncLayerNorm_{node.name}",
                )

                nodes_to_insert = [func_ln_node]

                # Add scale multiplication if non-trivial (not all ones)
                scale_data = model.get_initializer(scale)
                if scale_data is not None and not np.allclose(scale_data, 1.0):
                    scale_intermediate = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, act_shape
                    )
                    graph.value_info.append(scale_intermediate)
                    func_ln_node.output[0] = scale_intermediate.name

                    mul_node = oh.make_node("Mul", [scale_intermediate.name, scale], [act_out])
                    nodes_to_insert.append(mul_node)
                    model.set_tensor_datatype(scale_intermediate.name, idt)

                    last_node = mul_node
                else:
                    last_node = func_ln_node

                # Add bias if non-trivial (not all zeros)
                if bias is not None:
                    bias_data = model.get_initializer(bias)
                    if bias_data is not None and not np.allclose(bias_data, 0.0):
                        bias_intermediate = oh.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            act_shape
                        )
                        graph.value_info.append(bias_intermediate)
                        last_node.output[0] = bias_intermediate.name

                        add_node = oh.make_node("Add", [bias_intermediate.name, bias], [act_out])
                        nodes_to_insert.append(add_node)
                        model.set_tensor_datatype(bias_intermediate.name, wdt)

                replacements.append((node_idx, node, nodes_to_insert))

        # Apply all replacements if any were found
        if replacements:
            # Add opset import only when LayerNorm nodes were found
            existing_domains = {op.domain for op in model.model.opset_import}
            if "brainsmith.operators.general" not in existing_domains:
                model.model.opset_import.append(oh.make_opsetid("brainsmith.operators.general", 1))

            # Apply replacements in reverse order to maintain indices
            for node_idx, old_node, new_nodes in reversed(replacements):
                graph.node.remove(old_node)
                for i, new_node in enumerate(new_nodes):
                    graph.node.insert(node_idx + i, new_node)

            graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)
