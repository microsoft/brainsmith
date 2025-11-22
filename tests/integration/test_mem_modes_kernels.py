"""Integration tests for mem_modes with Thresholding and ElementwiseBinaryOp kernels.

Tests the full flow:
- Schema definition with mem_modes
- Design space building
- Design point instantiation
- Interface mem_mode access in kernel implementations
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor

from brainsmith.dataflow.builder import BuildContext, DesignSpaceBuilder
from brainsmith.kernels.thresholding.thresholding import (
    THRESHOLDING_SCHEMA,
    Thresholding,
)


def _create_param_getter(node):
    """Create a param_getter function for testing."""

    def param_getter(key):
        for attr in node.attribute:
            if attr.name == key:
                # Return the first value from the attribute
                if attr.HasField("i"):
                    return attr.i
                elif attr.HasField("f"):
                    return attr.f
                elif attr.HasField("s"):
                    return attr.s.decode("utf-8")
                elif attr.ints:
                    return list(attr.ints)
        raise KeyError(f"Attribute {key} not found")

    return param_getter


def _create_param_setter():
    """Create a param_setter function for testing (no-op)."""

    def param_setter(key, value):
        pass  # No-op for tests

    return param_setter


class TestThresholdingMemModes:
    """Integration tests for Thresholding kernel with mem_modes."""

    def test_thresholding_schema_has_mem_modes(self):
        """Test that Thresholding schema has mem_modes on thresholds input."""
        # Find thresholds input (index 1)
        thresholds_input = THRESHOLDING_SCHEMA.inputs[1]
        assert thresholds_input.name == "thresholds"
        assert thresholds_input.mem_modes is not None
        # mem_modes is now a static frozenset defining capabilities
        assert thresholds_input.mem_modes == frozenset({"embedded", "decoupled", "dynamic"})

    def test_thresholding_generates_input1_memtype(self):
        """Test that Thresholding design space has input1MemType parameter."""
        # Create a simple thresholding model
        import onnx

        inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 4])
        thresh = onnx.helper.make_tensor_value_info("thresh", onnx.TensorProto.FLOAT, [4, 1])
        out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 4])

        # Create threshold initializer
        threshold_values = np.array([[0.5], [1.0], [1.5], [2.0]], dtype=np.float32)
        thresh_init = onnx.helper.make_tensor(
            "thresh", onnx.TensorProto.FLOAT, [4, 1], threshold_values.flatten()
        )

        node = onnx.helper.make_node(
            "MultiThreshold",
            inputs=["inp", "thresh"],
            outputs=["out"],
            domain="qonnx.custom_op.general",
            name="threshold_node",
        )

        graph = onnx.helper.make_graph(
            [node], "threshold_graph", [inp, thresh], [out], initializer=[thresh_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("inp", DataType["INT8"])
        model_w.set_tensor_datatype("thresh", DataType["INT8"])
        model_w.set_tensor_datatype("out", DataType["INT8"])

        # Build design space
        node = model_w.graph.node[0]
        # Mark thresholds as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))
        build_ctx = BuildContext(
            schema=THRESHOLDING_SCHEMA,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # Verify input1MemType parameter exists
        assert "input1MemType" in design_space.parameters
        mem_modes = design_space.parameters["input1MemType"]

        # Should have all modes from static schema
        assert mem_modes == frozenset({"embedded", "decoupled", "dynamic"})

    def test_thresholding_mlo_forces_dynamic(self):
        """Test that adapt_for_loop_body() forces input1MemType to dynamic."""
        import onnx
        from qonnx.custom_op.registry import getCustomOp
        from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType

        inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 4])
        thresh = onnx.helper.make_tensor_value_info("thresh", onnx.TensorProto.FLOAT, [4, 1])
        out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 4])

        threshold_values = np.array([[0.5], [1.0], [1.5], [2.0]], dtype=np.float32)
        thresh_init = onnx.helper.make_tensor(
            "thresh", onnx.TensorProto.FLOAT, [4, 1], threshold_values.flatten()
        )

        # Create Thresholding node (not MultiThreshold)
        node = onnx.helper.make_node(
            "Thresholding",
            inputs=["inp", "thresh"],
            outputs=["out"],
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name="threshold_node",
            num_steps=1,
            act_val=0,
            num_input_vectors=[1],
            runtime_writeable_weights=0,
            PE=4,
        )

        # Mark thresholds as weight with initial mode "embedded"
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))

        graph = onnx.helper.make_graph(
            [node], "threshold_graph", [inp, thresh], [out], initializer=[thresh_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("inp", DataType["INT8"])
        model_w.set_tensor_datatype("thresh", DataType["INT8"])
        model_w.set_tensor_datatype("out", DataType["INT8"])

        # Get KernelOp instance and verify initial state
        node = model_w.graph.node[0]
        thl_inst = getCustomOp(node)
        assert thl_inst.get_nodeattr("input1MemType") == "embedded"

        # Call adapt_for_loop_body with MLO signature (thresholds are PARAMETER)
        loop_signature = [LoopBodyInputType.ACTIVATION, LoopBodyInputType.PARAMETER]
        thl_inst.adapt_for_loop_body(loop_signature)

        # Should be forced to dynamic
        assert thl_inst.get_nodeattr("input1MemType") == "dynamic"

    def test_thresholding_mlo_no_change_without_parameter(self):
        """Test that adapt_for_loop_body() doesn't change mode if not PARAMETER."""
        import onnx
        from qonnx.custom_op.registry import getCustomOp
        from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType

        inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 4])
        thresh = onnx.helper.make_tensor_value_info("thresh", onnx.TensorProto.FLOAT, [4, 1])
        out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 4])

        threshold_values = np.array([[0.5], [1.0], [1.5], [2.0]], dtype=np.float32)
        thresh_init = onnx.helper.make_tensor(
            "thresh", onnx.TensorProto.FLOAT, [4, 1], threshold_values.flatten()
        )

        node = onnx.helper.make_node(
            "Thresholding",
            inputs=["inp", "thresh"],
            outputs=["out"],
            domain="brainsmith.kernels",
            backend="fpgadataflow",
            name="threshold_node",
            num_steps=1,
            act_val=0,
            num_input_vectors=[1],
            runtime_writeable_weights=0,
            PE=4,
        )

        # Mark thresholds as weight with initial mode "embedded"
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))

        graph = onnx.helper.make_graph(
            [node], "threshold_graph", [inp, thresh], [out], initializer=[thresh_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("inp", DataType["INT8"])
        model_w.set_tensor_datatype("thresh", DataType["INT8"])
        model_w.set_tensor_datatype("out", DataType["INT8"])

        node = model_w.graph.node[0]
        thl_inst = getCustomOp(node)
        assert thl_inst.get_nodeattr("input1MemType") == "embedded"

        # Call adapt_for_loop_body with signature where thresholds are CONSTANT (not streamed)
        loop_signature = [LoopBodyInputType.ACTIVATION, LoopBodyInputType.CONSTANT]
        thl_inst.adapt_for_loop_body(loop_signature)

        # Should remain embedded (not changed to dynamic)
        assert thl_inst.get_nodeattr("input1MemType") == "embedded"

    def test_thresholding_interface_mem_mode_accessible(self):
        """Test that mem_mode is accessible from design point interface."""
        import onnx

        inp = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 4])
        thresh = onnx.helper.make_tensor_value_info("thresh", onnx.TensorProto.FLOAT, [4, 1])
        out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 4])

        threshold_values = np.array([[0.5], [1.0], [1.5], [2.0]], dtype=np.float32)
        thresh_init = onnx.helper.make_tensor(
            "thresh", onnx.TensorProto.FLOAT, [4, 1], threshold_values.flatten()
        )

        node = onnx.helper.make_node(
            "MultiThreshold",
            inputs=["inp", "thresh"],
            outputs=["out"],
            domain="qonnx.custom_op.general",
            name="threshold_node",
        )

        graph = onnx.helper.make_graph(
            [node], "threshold_graph", [inp, thresh], [out], initializer=[thresh_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("inp", DataType["INT8"])
        model_w.set_tensor_datatype("thresh", DataType["INT8"])
        model_w.set_tensor_datatype("out", DataType["INT8"])

        # Build design space
        node = model_w.graph.node[0]
        # Mark thresholds as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))
        build_ctx = BuildContext(
            schema=THRESHOLDING_SCHEMA,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # Configure with embedded mode
        design_point = design_space.configure({"PE": 1, "input1MemType": "embedded"})

        # Verify mem_mode is accessible from interface
        thresholds_iface = design_point.inputs["thresholds"]
        assert thresholds_iface.mem_mode == "embedded"
        assert thresholds_iface.is_weight is True

        # Configure with decoupled mode
        design_point2 = design_space.configure({"PE": 1, "input1MemType": "decoupled"})
        assert design_point2.inputs["thresholds"].mem_mode == "decoupled"


class TestElementwiseBinaryOpMemModes:
    """Integration tests for ElementwiseBinaryOp kernel with mem_modes."""

    def test_elementwise_schema_has_mem_modes(self):
        """Test that ElementwiseBinaryOp schema has mem_modes on RHS input."""
        from brainsmith.kernels.elementwise_binary.elementwise_binary import (
            ELEMENTWISE_BINARY_SCHEMA,
        )

        # Find RHS input (index 1)
        rhs_input = ELEMENTWISE_BINARY_SCHEMA.inputs[1]
        assert rhs_input.name == "rhs"
        assert rhs_input.mem_modes is not None
        # mem_modes is now a static frozenset defining capabilities
        assert rhs_input.mem_modes == frozenset({"embedded", "decoupled", "dynamic"})

    def test_elementwise_generates_input1_memtype(self):
        """Test that ElementwiseBinaryOp design space has input1MemType parameter."""
        from brainsmith.kernels.elementwise_binary.elementwise_binary import (
            ELEMENTWISE_BINARY_SCHEMA,
        )
        import onnx

        # Create simple Add operation with static RHS
        lhs = onnx.helper.make_tensor_value_info("lhs", onnx.TensorProto.FLOAT, [4])
        rhs = onnx.helper.make_tensor_value_info("rhs", onnx.TensorProto.FLOAT, [4])
        out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [4])

        # RHS is static (initializer)
        rhs_data = gen_finn_dt_tensor(DataType["INT8"], [4])
        rhs_init = onnx.numpy_helper.from_array(rhs_data, name="rhs")

        node = onnx.helper.make_node(
            "Add",
            inputs=["lhs", "rhs"],
            outputs=["out"],
            name="add_node",
            func="Add",
            input_pattern="dynamic_static",
        )

        graph = onnx.helper.make_graph([node], "add_graph", [lhs, rhs], [out], initializer=[rhs_init])
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("lhs", DataType["INT8"])
        model_w.set_tensor_datatype("rhs", DataType["INT8"])
        model_w.set_tensor_datatype("out", DataType["INT8"])

        # Build design space
        node = model_w.graph.node[0]
        # Mark RHS as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))
        build_ctx = BuildContext(
            schema=ELEMENTWISE_BINARY_SCHEMA,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # Verify input1MemType parameter exists (RHS is index 1)
        assert "input1MemType" in design_space.parameters
        mem_modes = design_space.parameters["input1MemType"]
        # Should have all modes from static schema
        assert mem_modes == frozenset({"embedded", "decoupled", "dynamic"})

    def test_elementwise_no_mem_mode_for_dynamic_lhs(self):
        """Test that LHS (dynamic input) does not have mem_mode parameter."""
        from brainsmith.kernels.elementwise_binary.elementwise_binary import (
            ELEMENTWISE_BINARY_SCHEMA,
        )
        import onnx

        lhs = onnx.helper.make_tensor_value_info("lhs", onnx.TensorProto.FLOAT, [4])
        rhs = onnx.helper.make_tensor_value_info("rhs", onnx.TensorProto.FLOAT, [4])
        out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [4])

        rhs_data = gen_finn_dt_tensor(DataType["INT8"], [4])
        rhs_init = onnx.numpy_helper.from_array(rhs_data, name="rhs")

        node = onnx.helper.make_node(
            "Add",
            inputs=["lhs", "rhs"],
            outputs=["out"],
            name="add_node",
            func="Add",
            input_pattern="dynamic_static",
        )

        graph = onnx.helper.make_graph([node], "add_graph", [lhs, rhs], [out], initializer=[rhs_init])
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("lhs", DataType["INT8"])
        model_w.set_tensor_datatype("rhs", DataType["INT8"])
        model_w.set_tensor_datatype("out", DataType["INT8"])

        # Build design space
        node = model_w.graph.node[0]
        # Mark RHS as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))
        build_ctx = BuildContext(
            schema=ELEMENTWISE_BINARY_SCHEMA,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # LHS should NOT have mem_mode parameter (it's dynamic)
        assert "input0MemType" not in design_space.parameters

        # But RHS should have it (it's a weight)
        assert "input1MemType" in design_space.parameters


class TestChannelwiseOpMemModesIntegration:
    """Integration tests for ChannelwiseOp kernel with mem_modes."""

    def test_channelwise_interface_mem_mode_accessible(self):
        """Test that mem_mode is accessible from design point interface."""
        from brainsmith.kernels.channelwise.channelwise import CHANNELWISE_SCHEMA
        import onnx

        # Create simple Add operation with static parameters
        lhs = onnx.helper.make_tensor_value_info("lhs", onnx.TensorProto.FLOAT, [4])
        params = onnx.helper.make_tensor_value_info("params", onnx.TensorProto.FLOAT, [4])
        out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [4])

        # Parameters are static (initializer)
        params_data = gen_finn_dt_tensor(DataType["INT8"], [4])
        params_init = onnx.numpy_helper.from_array(params_data, name="params")

        node = onnx.helper.make_node(
            "Add",
            inputs=["lhs", "params"],
            outputs=["out"],
            name="add_node",
            func="Add",
        )

        graph = onnx.helper.make_graph(
            [node], "add_graph", [lhs, params], [out], initializer=[params_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("lhs", DataType["INT8"])
        model_w.set_tensor_datatype("params", DataType["INT8"])
        model_w.set_tensor_datatype("out", DataType["INT8"])

        # Build design space
        node = model_w.graph.node[0]
        # Mark parameters as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))
        build_ctx = BuildContext(
            schema=CHANNELWISE_SCHEMA,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # Configure with embedded mode
        design_point = design_space.configure({
            "PE": 1,
            "input1MemType": "embedded",
            "ram_style": "distributed"
        })

        # Verify mem_mode is accessible from interface
        params_iface = design_point.inputs["parameters"]
        assert params_iface.mem_mode == "embedded"
        assert params_iface.is_weight is True

    def test_channelwise_no_mem_mode_for_dynamic_input(self):
        """Test that LHS (dynamic input) does not have mem_mode parameter."""
        from brainsmith.kernels.channelwise.channelwise import CHANNELWISE_SCHEMA
        import onnx

        lhs = onnx.helper.make_tensor_value_info("lhs", onnx.TensorProto.FLOAT, [4])
        params = onnx.helper.make_tensor_value_info("params", onnx.TensorProto.FLOAT, [4])
        out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [4])

        params_data = gen_finn_dt_tensor(DataType["INT8"], [4])
        params_init = onnx.numpy_helper.from_array(params_data, name="params")

        node = onnx.helper.make_node(
            "Add",
            inputs=["lhs", "params"],
            outputs=["out"],
            name="add_node",
            func="Add",
        )

        graph = onnx.helper.make_graph(
            [node], "add_graph", [lhs, params], [out], initializer=[params_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("lhs", DataType["INT8"])
        model_w.set_tensor_datatype("params", DataType["INT8"])
        model_w.set_tensor_datatype("out", DataType["INT8"])

        # Build design space
        node = model_w.graph.node[0]
        # Mark parameters as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))
        build_ctx = BuildContext(
            schema=CHANNELWISE_SCHEMA,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # LHS should NOT have mem_mode parameter (it's dynamic)
        assert "input0MemType" not in design_space.parameters

        # But parameters should have it (static weight)
        assert "input1MemType" in design_space.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
