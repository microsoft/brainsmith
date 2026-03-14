"""Unit tests for mem_modes interface-level DSE parameter system.

Tests cover:
- Schema validation of mem_modes
- Builder generation of input<idx>MemType parameters
- Callable mem_modes for MLO filtering
- InterfaceDesignPoint.mem_mode population
- Integration with Thresholding and ElementwiseBinaryOp
"""

import numpy as np
import onnx
import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor

import brainsmith.dataflow as df
from brainsmith.dataflow.builder import BuildContext, DesignSpaceBuilder
from brainsmith.dataflow.schemas import InputSchema, KernelSchema, OutputSchema
from brainsmith.dataflow.types import FULL_DIM


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


class TestMemModesSchemaValidation:
    """Test InputSchema validation of mem_modes."""

    def test_valid_mem_modes_frozenset(self):
        """Test that valid mem_modes frozenset is accepted."""
        schema = InputSchema(
            name="test_input",
            block_tiling=[FULL_DIM],
            stream_tiling=[],
            mem_modes=frozenset({"embedded", "decoupled", "dynamic"}),
        )
        assert schema.mem_modes == frozenset({"embedded", "decoupled", "dynamic"})

    def test_valid_mem_modes_callable(self):
        """Test that callable mem_modes is accepted."""

        def compute_modes(ctx):
            return frozenset({"embedded"})

        schema = InputSchema(
            name="test_input",
            block_tiling=[FULL_DIM],
            stream_tiling=[],
            mem_modes=compute_modes,
        )
        assert callable(schema.mem_modes)

    def test_invalid_mem_modes_type(self):
        """Test that non-frozenset/callable mem_modes raises TypeError."""
        with pytest.raises(TypeError, match="must be frozenset or callable"):
            InputSchema(
                name="test_input",
                block_tiling=[FULL_DIM],
                stream_tiling=[],
                mem_modes={"embedded", "decoupled"},  # set, not frozenset
            )

    def test_invalid_mem_mode_values(self):
        """Test that invalid mode names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mem_modes"):
            InputSchema(
                name="test_input",
                block_tiling=[FULL_DIM],
                stream_tiling=[],
                mem_modes=frozenset({"embedded", "invalid_mode"}),
            )

    def test_mem_modes_none_is_valid(self):
        """Test that mem_modes=None (non-weight input) is valid."""
        schema = InputSchema(
            name="test_input",
            block_tiling=[FULL_DIM],
            stream_tiling=["PE"],
            mem_modes=None,  # Dynamic input, not a weight
        )
        assert schema.mem_modes is None


class TestBuilderParameterGeneration:
    """Test that builder generates input<idx>MemType parameters from mem_modes."""

    def test_generates_input0_memtype_parameter(self):
        """Test that mem_modes on input0 generates input0MemType parameter."""
        # Create minimal schema with mem_modes on first input
        schema = KernelSchema(
            name="TestKernel",
            inputs=[
                InputSchema(
                    name="weights",
                    block_tiling=[],
                    stream_tiling=[],
                    mem_modes=frozenset({"embedded", "decoupled"}),
                ),
            ],
            outputs=[
                OutputSchema(
                    name="output",
                    block_tiling=[FULL_DIM],
                    stream_tiling=[],
                ),
            ],
            kernel_params={},
            dse_parameters={},
        )

        # Create minimal ONNX model
        import onnx

        inp = onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, [4])
        out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])
        weight_data = gen_finn_dt_tensor(DataType["INT8"], [4])
        weight_init = onnx.numpy_helper.from_array(weight_data, name="weights")
        node = onnx.helper.make_node("TestOp", inputs=["weights"], outputs=["output"], name="test_node")
        graph = onnx.helper.make_graph(
            [node], "test_graph", [inp], [out], initializer=[weight_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)

        # Build design space
        node = model_w.graph.node[0]
        # Mark weights as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input0MemType", "embedded"))
        build_ctx = BuildContext(
            schema=schema,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # Verify input0MemType parameter exists
        assert "input0MemType" in design_space.parameters
        assert design_space.parameters["input0MemType"] == frozenset({"embedded", "decoupled"})

    def test_generates_input1_memtype_parameter(self):
        """Test that mem_modes on input1 generates input1MemType parameter."""
        schema = KernelSchema(
            name="TestKernel",
            inputs=[
                InputSchema(
                    name="data",
                    block_tiling=[FULL_DIM],
                    stream_tiling=["PE"],
                    mem_modes=None,  # Dynamic input
                ),
                InputSchema(
                    name="thresholds",
                    block_tiling=[],
                    stream_tiling=[],
                    mem_modes=frozenset({"embedded", "decoupled", "dynamic"}),
                ),
            ],
            outputs=[
                OutputSchema(
                    name="output",
                    block_tiling=[FULL_DIM],
                    stream_tiling=["PE"],
                ),
            ],
            kernel_params={},
            dse_parameters={"PE": df.ParameterSpec(name="PE", values=[1, 2, 4], default=1)},
        )

        # Create ONNX model with initializer for thresholds
        import onnx

        inp1 = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [16])
        inp2 = onnx.helper.make_tensor_value_info("thresholds", onnx.TensorProto.FLOAT, [16])
        out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [16])
        threshold_data = gen_finn_dt_tensor(DataType["INT8"], [16])
        threshold_init = onnx.numpy_helper.from_array(threshold_data, name="thresholds")
        node = onnx.helper.make_node(
            "TestOp", inputs=["data", "thresholds"], outputs=["output"], name="test_node"
        )
        graph = onnx.helper.make_graph(
            [node], "test_graph", [inp1, inp2], [out], initializer=[threshold_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)

        # Build design space
        node = model_w.graph.node[0]
        # Mark thresholds as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input1MemType", "embedded"))
        build_ctx = BuildContext(
            schema=schema,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # Verify input1MemType parameter exists (thresholds is index 1)
        assert "input1MemType" in design_space.parameters
        assert design_space.parameters["input1MemType"] == frozenset(
            {"embedded", "decoupled", "dynamic"}
        )

        # Verify input0 does NOT have mem_mode parameter (dynamic input)
        assert "input0MemType" not in design_space.parameters


class TestCallableMemModes:
    """Test callable mem_modes for context-aware filtering."""

    def test_callable_mlo_filtering(self):
        """Test that callable filters to dynamic mode when mlo_max_iter > 1."""

        def compute_modes(ctx: BuildContext) -> frozenset[str]:
            """Filter modes based on MLO context."""
            try:
                mlo_max_iter = ctx.param_getter("mlo_max_iter")
                if mlo_max_iter and mlo_max_iter > 1:
                    return frozenset({"dynamic"})  # MLO forces streaming
            except (AttributeError, KeyError):
                pass
            return frozenset({"embedded", "decoupled"})

        schema = KernelSchema(
            name="TestKernel",
            inputs=[
                InputSchema(
                    name="weights",
                    block_tiling=[],
                    stream_tiling=[],
                    mem_modes=compute_modes,  # Callable
                ),
            ],
            outputs=[
                OutputSchema(
                    name="output",
                    block_tiling=[FULL_DIM],
                    stream_tiling=[],
                ),
            ],
            kernel_params={"mlo_max_iter": ("i", False, 1)},
            dse_parameters={},
        )

        # Test 1: Non-MLO context (mlo_max_iter=1)
        import onnx

        inp = onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, [4])
        out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])
        weight_data = gen_finn_dt_tensor(DataType["INT8"], [4])
        weight_init = onnx.numpy_helper.from_array(weight_data, name="weights")
        node = onnx.helper.make_node(
            "TestOp",
            inputs=["weights"],
            outputs=["output"],
            name="test_node",
            mlo_max_iter=1,  # Non-MLO
        )
        graph = onnx.helper.make_graph(
            [node], "test_graph", [inp], [out], initializer=[weight_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)

        node = model_w.graph.node[0]
        # Mark weights as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input0MemType", "embedded"))
        build_ctx = BuildContext(
            schema=schema,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # Should have embedded and decoupled
        assert design_space.parameters["input0MemType"] == frozenset({"embedded", "decoupled"})

        # Test 2: MLO context (mlo_max_iter=4)
        node_mlo = onnx.helper.make_node(
            "TestOp",
            inputs=["weights"],
            outputs=["output"],
            name="test_node_mlo",
            mlo_max_iter=4,  # MLO mode!
        )
        # Reuse the weight_init from above
        graph_mlo = onnx.helper.make_graph(
            [node_mlo], "test_graph", [inp], [out], initializer=[weight_init]
        )
        model_mlo = onnx.helper.make_model(graph_mlo)
        model_w_mlo = ModelWrapper(model_mlo)

        node_mlo = model_w_mlo.graph.node[0]
        # Mark weights as weight
        node_mlo.attribute.append(onnx.helper.make_attribute("input0MemType", "embedded"))
        build_ctx_mlo = BuildContext(
            schema=schema,
            model_w=model_w_mlo,
            node=node_mlo,
            param_getter=_create_param_getter(node_mlo),
            param_setter=_create_param_setter(),
        )
        design_space_mlo = DesignSpaceBuilder().build(build_ctx_mlo)

        # Should only have dynamic mode
        assert design_space_mlo.parameters["input0MemType"] == frozenset({"dynamic"})


class TestInterfaceDesignPointMemMode:
    """Test that InterfaceDesignPoint.mem_mode is populated from config."""

    def test_mem_mode_populated_on_instantiation(self):
        """Test that mem_mode is extracted from params and set on interface."""
        schema = KernelSchema(
            name="TestKernel",
            inputs=[
                InputSchema(
                    name="weights",
                    block_tiling=[],
                    stream_tiling=[],
                    mem_modes=frozenset({"embedded", "decoupled"}),
                ),
            ],
            outputs=[
                OutputSchema(
                    name="output",
                    block_tiling=[FULL_DIM],
                    stream_tiling=[],
                ),
            ],
            kernel_params={},
            dse_parameters={},
        )

        # Create ONNX model
        import onnx

        inp = onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, [4])
        out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])
        weight_data = gen_finn_dt_tensor(DataType["INT8"], [4])
        weight_init = onnx.numpy_helper.from_array(weight_data, name="weights")
        node = onnx.helper.make_node("TestOp", inputs=["weights"], outputs=["output"], name="test_node")
        graph = onnx.helper.make_graph(
            [node], "test_graph", [inp], [out], initializer=[weight_init]
        )
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)

        # Build design space
        node = model_w.graph.node[0]
        # Mark weights as weight (simulating what InferKernel would do)
        node.attribute.append(onnx.helper.make_attribute("input0MemType", "embedded"))
        build_ctx = BuildContext(
            schema=schema,
            model_w=model_w,
            node=node,
            param_getter=_create_param_getter(node),
            param_setter=_create_param_setter(),
        )
        design_space = DesignSpaceBuilder().build(build_ctx)

        # Configure with embedded mode
        design_point = design_space.configure({"input0MemType": "embedded"})

        # Verify mem_mode is set on interface
        assert design_point.inputs["weights"].mem_mode == "embedded"

        # Configure with decoupled mode
        design_point2 = design_space.configure({"input0MemType": "decoupled"})
        assert design_point2.inputs["weights"].mem_mode == "decoupled"


class TestChannelwiseOpMemModes:
    """Test mem_modes for ChannelwiseOp kernel."""

    def test_channelwise_schema_has_mem_modes(self):
        """Test that ChannelwiseOp schema has mem_modes on parameters input."""
        from brainsmith.kernels.channelwise.channelwise import CHANNELWISE_SCHEMA

        # Find parameters input (index 1)
        params_input = CHANNELWISE_SCHEMA.inputs[1]
        assert params_input.name == "parameters"
        assert params_input.mem_modes is not None
        assert params_input.mem_modes == frozenset({"embedded"})

    def test_channelwise_generates_input1_memtype(self):
        """Test that ChannelwiseOp design space has input1MemType parameter."""
        from brainsmith.kernels.channelwise.channelwise import CHANNELWISE_SCHEMA
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
        )

        graph = onnx.helper.make_graph([node], "add_graph", [lhs, rhs], [out], initializer=[rhs_init])
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("lhs", DataType["INT8"])
        model_w.set_tensor_datatype("rhs", DataType["INT8"])
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

        # Verify input1MemType parameter exists (RHS is index 1)
        assert "input1MemType" in design_space.parameters
        mem_modes = design_space.parameters["input1MemType"]
        assert mem_modes == frozenset({"embedded"})

    def test_channelwise_interface_mem_mode_accessible(self):
        """Test that mem_mode is accessible from design point interface."""
        from brainsmith.kernels.channelwise.channelwise import CHANNELWISE_SCHEMA
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
        )

        graph = onnx.helper.make_graph([node], "add_graph", [lhs, rhs], [out], initializer=[rhs_init])
        model = onnx.helper.make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("lhs", DataType["INT8"])
        model_w.set_tensor_datatype("rhs", DataType["INT8"])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
