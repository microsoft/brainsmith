# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable test helpers for kernel unit testing.

This module provides builders for common ONNX test models used in
kernel schema validation, inference testing, and transformation testing.

For parity testing (comparing manual vs auto implementations), use
tests/parity/test_fixtures.py and ParityTestBase instead.

Key Features:
- Fluent OnnxModelBuilder API for constructing test models
- Convenience functions for common patterns (binary, unary, parametric ops)
- Specialized helpers for complex kernels (VVAU, MultiThreshold, DuplicateStreams)
- Eliminates 100+ lines of boilerplate per kernel test file
- Single source of truth for test model construction

Basic Usage:
    from tests.fixtures.kernel_test_helpers import make_parametric_op_model

    # Simple parametric operation (one dynamic, one static input)
    model, node = make_parametric_op_model("Add", param_input="bias")
    assert ChannelwiseOp.can_infer_from(node, model)

    # Binary operation (two dynamic inputs)
    model, node = make_binary_op_model("Add")
    assert AddStreams.can_infer_from(node, model)

Advanced Usage:
    # MultiThreshold with auto-computed thresholds
    model, node = make_multithreshold_model(
        shape=[1, 28, 28, 128],
        output_dtype="UINT4"  # Auto-computes 15 thresholds
    )

    # VVAU with depthwise sparse weights
    model, node = make_vvau_model(
        channels=16,
        kernel_shape=[3, 3],
        mode="vvau_node"
    )

    # Broadcasting operations
    model, node = make_broadcast_model(
        [1,64,64,128], [128], "Add"  # Channel broadcasting
    )

    # Custom builder with fluent API
    model, node = (OnnxModelBuilder()
        .op_type("Add")
        .inputs(["data", "bias"])
        .shape([1, 8, 8, 64])
        .static_input("bias", shape=[64])
        .datatype(DataType["INT8"])
        .build())
"""


import numpy as np
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model


class OnnxModelBuilder:
    """Fluent builder for ONNX test models with sane defaults.

    NOTE: OnnxModelBuilder is primarily for rapid prototyping and exploration.
    Production tests typically use kernel-specific helpers located in
    brainsmith/kernels/*/tests/ directories.

    Eliminates boilerplate for common kernel testing patterns:
    - Binary ops (Add, Mul) with two dynamic inputs
    - Unary ops (Softmax, LayerNorm) with one dynamic input
    - Parametric ops (Channelwise) with dynamic + static inputs
    - Comparison ops (LessOrEqual, GreaterOrEqual)

    Example:
        # Simple Add with two inputs
        model, node = (OnnxModelBuilder()
            .op_type("Add")
            .inputs(["in0", "in1"])
            .shape([1, 224, 224, 64])
            .datatype(DataType["INT8"])
            .build())

        # Parametric op with static parameter
        model, node = (OnnxModelBuilder()
            .op_type("Add")
            .inputs(["data", "bias"])
            .shape([1, 8, 8, 64])
            .static_input("bias", shape=[64])
            .datatype(DataType["INT8"])
            .build())

        # Swapped input order (for testing canonical ordering)
        model, node = (OnnxModelBuilder()
            .op_type("Add")
            .inputs(["bias", "data"])  # Reversed!
            .shape([1, 8, 8, 64])
            .static_input("bias", shape=[64])
            .build())
    """

    def __init__(self):
        """Initialize builder with sensible defaults."""
        self._op_type: str = "Add"
        self._inputs: list[str] = ["in0", "in1"]
        self._outputs: list[str] = ["output"]
        self._shape: list[int] = [1, 8, 8, 64]
        self._datatype: DataType = DataType["INT8"]
        self._static_inputs: dict = {}  # {name: shape}
        self._initializer_values: dict = {}  # {name: numpy_array}
        self._input_shapes: dict = {}  # Override per-input shapes
        self._domain: str = ""  # Standard ONNX domain
        self._node_name: str = None

    def op_type(self, op_type: str) -> "OnnxModelBuilder":
        """Set operation type (e.g., 'Add', 'Softmax').

        Args:
            op_type: ONNX operation type string

        Returns:
            Self for method chaining
        """
        self._op_type = op_type
        return self

    def inputs(self, inputs: list[str]) -> "OnnxModelBuilder":
        """Set input tensor names.

        Args:
            inputs: List of input tensor names

        Returns:
            Self for method chaining
        """
        self._inputs = inputs
        return self

    def outputs(self, outputs: list[str]) -> "OnnxModelBuilder":
        """Set output tensor names.

        Args:
            outputs: List of output tensor names

        Returns:
            Self for method chaining
        """
        self._outputs = outputs
        return self

    def shape(self, shape: list[int]) -> "OnnxModelBuilder":
        """Set default shape for all inputs/outputs.

        Args:
            shape: Tensor shape (e.g., [1, 8, 8, 64])

        Returns:
            Self for method chaining
        """
        self._shape = shape
        return self

    def input_shape(self, name: str, shape: list[int]) -> "OnnxModelBuilder":
        """Override shape for specific input.

        Args:
            name: Input tensor name
            shape: Shape for this specific input

        Returns:
            Self for method chaining
        """
        self._input_shapes[name] = shape
        return self

    def datatype(self, datatype: DataType) -> "OnnxModelBuilder":
        """Set datatype for all tensors.

        Args:
            datatype: QONNX DataType (e.g., DataType["INT8"])

        Returns:
            Self for method chaining
        """
        self._datatype = datatype
        return self

    def static_input(
        self, name: str, shape: list[int] | int = None, values: np.ndarray = None
    ) -> "OnnxModelBuilder":
        """Mark input as static (initializer) with optional shape/values.

        Args:
            name: Input tensor name to make static
            shape: Shape for the static input (int or list). If int, treated as [int].
                  If None, uses last dimension of default shape (for per-channel params)
            values: Explicit numpy array. If None, uses zeros with appropriate shape.

        Returns:
            Self for method chaining

        Example:
            # Per-channel parameter (inferred from default shape)
            builder.static_input("bias")

            # Scalar parameter
            builder.static_input("scalar", shape=1)

            # Custom shape
            builder.static_input("weight", shape=[64, 32])

            # Custom values
            builder.static_input("threshold", values=np.array([0.5]))
        """
        if values is not None:
            self._initializer_values[name] = values
            self._static_inputs[name] = list(values.shape)
        else:
            if shape is None:
                # Default: per-channel parameter (last dim of default shape)
                shape = [self._shape[-1]]
            elif isinstance(shape, int):
                shape = [shape]

            self._static_inputs[name] = shape
            self._initializer_values[name] = np.zeros(shape)

        return self

    def domain(self, domain: str) -> "OnnxModelBuilder":
        """Set custom domain (default is standard ONNX).

        Args:
            domain: ONNX domain string (e.g., "brainsmith.kernels")

        Returns:
            Self for method chaining
        """
        self._domain = domain
        return self

    def name(self, name: str) -> "OnnxModelBuilder":
        """Set node name.

        Args:
            name: Node name string

        Returns:
            Self for method chaining
        """
        self._node_name = name
        return self

    def build(self) -> tuple[ModelWrapper, NodeProto]:
        """Build the ONNX model and return (model, node).

        Returns:
            (ModelWrapper, NodeProto): The model and the target operation node

        Raises:
            ValueError: If configuration is invalid
        """
        # Use QONNX convention (FLOAT containers with DataType annotations)
        from tests.fixtures.model_annotation import tensorproto_for_datatype

        tensorproto_type = tensorproto_for_datatype(self._datatype)

        # Create input value infos (non-static only)
        dynamic_inputs = []
        for inp_name in self._inputs:
            if inp_name not in self._static_inputs:
                shape = self._input_shapes.get(inp_name, self._shape)
                dynamic_inputs.append(
                    helper.make_tensor_value_info(inp_name, tensorproto_type, shape)
                )

        # Create initializers for static inputs
        initializers = []
        for name, shape in self._static_inputs.items():
            values = self._initializer_values.get(name, np.zeros(shape))
            init = helper.make_tensor(name, tensorproto_type, shape, values.flatten().tolist())
            initializers.append(init)

        # Create output value info
        output_shape = self._shape  # Could be customized if needed
        outputs_info = [
            helper.make_tensor_value_info(out, tensorproto_type, output_shape)
            for out in self._outputs
        ]

        # Create node
        node_args = {
            "inputs": self._inputs,
            "outputs": self._outputs,
        }
        if self._node_name:
            node_args["name"] = self._node_name
        if self._domain:
            node_args["domain"] = self._domain

        node = helper.make_node(self._op_type, **node_args)

        # Create graph
        graph = helper.make_graph([node], "test_graph", dynamic_inputs, outputs_info, initializers)

        # Create model
        model_proto = qonnx_make_model(graph)
        model = ModelWrapper(model_proto)

        # Set datatypes for all tensors
        for inp_name in self._inputs:
            model.set_tensor_datatype(inp_name, self._datatype)
        for out in self._outputs:
            model.set_tensor_datatype(out, self._datatype)

        return model, node


# =============================================================================
# Convenience Functions for Common Patterns
# =============================================================================


def make_binary_op_model(
    op_type: str,
    shape: list[int] = None,
    input0: str = "in0",
    input1: str = "in1",
    output: str = "output",
    datatype: DataType = DataType["INT8"],
) -> tuple[ModelWrapper, NodeProto]:
    """Create binary operation model (Add, Mul, etc.) with two dynamic inputs.

    Use this for kernels that operate on two streaming inputs, such as:
    - AddStreams (element-wise add of two streams)
    - MulStreams (element-wise multiply of two streams)

    Args:
        op_type: Operation type (e.g., "Add", "Mul")
        shape: Input/output shape (default: [1, 8, 8, 64])
        input0: First input name (default: "in0")
        input1: Second input name (default: "in1")
        output: Output name (default: "output")
        datatype: Tensor datatype (default: DataType["INT8"])

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> model, node = make_binary_op_model("Add")
        >>> assert AddStreams.can_infer_from(node, model)

        >>> # Custom shape and names
        >>> model, node = make_binary_op_model(
        ...     "Add",
        ...     shape=[1, 224, 224, 64],
        ...     input0="stream0",
        ...     input1="stream1"
        ... )
    """
    builder = (
        OnnxModelBuilder()
        .op_type(op_type)
        .inputs([input0, input1])
        .outputs([output])
        .datatype(datatype)
    )

    if shape:
        builder.shape(shape)

    return builder.build()


def make_parametric_op_model(
    op_type: str,
    dynamic_input: str = "data",
    param_input: str = "param",
    shape: list[int] = None,
    param_shape: list[int] | int = None,
    datatype: DataType = DataType["INT8"],
    input_order: str = "dynamic_first",
) -> tuple[ModelWrapper, NodeProto]:
    """Create parametric operation model (one dynamic, one static input).

    Use this for kernels that have one streaming input and one static parameter:
    - Channelwise operations (Add with bias, Mul with scale, comparison with threshold)
    - Thresholding operations
    - Any op where one input is weights/parameters

    Args:
        op_type: Operation type (e.g., "Add", "Mul", "LessOrEqual")
        dynamic_input: Name for dynamic (streaming) input (default: "data")
        param_input: Name for static (parameter) input (default: "param")
        shape: Shape for dynamic input (default: [1, 8, 8, 64])
        param_shape: Shape for parameter. If None, uses per-channel (last dim of shape).
                    If int, converts to [int]. If list, uses as-is.
        datatype: Tensor datatype (default: DataType["INT8"])
        input_order: "dynamic_first" or "static_first" (for testing canonical ordering)

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> # Canonical order (dynamic, static)
        >>> model, node = make_parametric_op_model("Add", param_input="bias")
        >>> assert ChannelwiseOp.can_infer_from(node, model)

        >>> # Swapped order (tests that inference handles this)
        >>> model, node = make_parametric_op_model(
        ...     "Add",
        ...     param_input="bias",
        ...     input_order="static_first"
        ... )

        >>> # Scalar parameter
        >>> model, node = make_parametric_op_model(
        ...     "Add",
        ...     param_input="scalar",
        ...     param_shape=1
        ... )

        >>> # Comparison operation
        >>> model, node = make_parametric_op_model(
        ...     "LessOrEqual",
        ...     param_input="threshold"
        ... )
    """
    if input_order == "dynamic_first":
        inputs = [dynamic_input, param_input]
    else:
        inputs = [param_input, dynamic_input]

    builder = (
        OnnxModelBuilder()
        .op_type(op_type)
        .inputs(inputs)
        .static_input(param_input, shape=param_shape)
        .datatype(datatype)
    )

    if shape:
        builder.shape(shape)

    return builder.build()


def make_unary_op_model(
    op_type: str,
    input_name: str = "input",
    output: str = "output",
    shape: list[int] = None,
    datatype: DataType = DataType["INT8"],
) -> tuple[ModelWrapper, NodeProto]:
    """Create unary operation model (Softmax, LayerNorm, etc.).

    Use this for kernels that have a single streaming input:
    - Softmax
    - LayerNorm
    - Activation functions (ReLU, Sigmoid, etc.)

    Args:
        op_type: Operation type (e.g., "Softmax", "LayerNorm")
        input_name: Input tensor name (default: "input")
        output: Output tensor name (default: "output")
        shape: Input/output shape (default: [1, 8, 8, 64])
        datatype: Tensor datatype (default: DataType["INT8"])

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> model, node = make_unary_op_model("Softmax", shape=[1, 10, 768])
        >>> assert SoftmaxOp.can_infer_from(node, model)

        >>> # LayerNorm with custom name
        >>> model, node = make_unary_op_model(
        ...     "LayerNorm",
        ...     input_name="activations",
        ...     shape=[1, 12, 768]
        ... )
    """
    builder = (
        OnnxModelBuilder()
        .op_type(op_type)
        .inputs([input_name])
        .outputs([output])
        .datatype(datatype)
    )

    if shape:
        builder.shape(shape)

    return builder.build()


# =============================================================================
# Specialized Helper Functions for Complex Kernel Patterns
# =============================================================================


def make_multithreshold_model(
    shape: list[int] = [1, 28, 28, 128],
    input_dtype: str = "INT8",
    threshold_dtype: str = "INT8",
    output_dtype: str = "UINT4",
    out_scale: float = 1.0,
    out_bias: int = 0,
    num_thresholds: int | None = None,
) -> tuple[ModelWrapper, NodeProto]:
    """Create MultiThreshold operation model with automatic threshold generation.

    Use this for thresholding operations that quantize activations to lower bitwidths.
    Automatically generates evenly-spaced threshold values and configures all
    MultiThreshold attributes.

    Args:
        shape: Input/output tensor shape (e.g., [1, 28, 28, 128])
        input_dtype: Input datatype (e.g., "INT8")
        threshold_dtype: Threshold parameter datatype (e.g., "INT8")
        output_dtype: Output datatype after thresholding (e.g., "UINT4")
        out_scale: Output scaling factor (default: 1.0)
        out_bias: Output bias/ActVal (default: 0)
        num_thresholds: Number of thresholds. If None, auto-computes from output_dtype
                       as 2^(bitwidth) - 1

    Returns:
        (model, node): ModelWrapper and MultiThreshold node

    Example:
        >>> # UINT4 quantization (15 thresholds)
        >>> model, node = make_multithreshold_model(
        ...     shape=[1, 28, 28, 128],
        ...     input_dtype="INT8",
        ...     output_dtype="UINT4"
        ... )
        >>> # Node has 15 thresholds per channel (2^4 - 1)

        >>> # Custom number of thresholds
        >>> model, node = make_multithreshold_model(
        ...     shape=[1, 28, 28, 128],
        ...     output_dtype="UINT8",
        ...     num_thresholds=100  # Override default 255
        ... )
    """
    channels = shape[-1]

    # Auto-compute number of thresholds if not provided
    if num_thresholds is None:
        output_dt = DataType[output_dtype]
        num_thresholds = (2 ** output_dt.bitwidth()) - 1

    # Create input/output tensor infos
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    thresh_shape = [channels, num_thresholds]
    helper.make_tensor_value_info("thresh", TensorProto.FLOAT, thresh_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    # Create MultiThreshold node with attributes
    node = helper.make_node(
        "MultiThreshold",
        inputs=["inp", "thresh"],
        outputs=["outp"],
        domain="qonnx.custom_op.general",
        name="MultiThreshold_0",
    )

    node.attribute.extend(
        [
            helper.make_attribute("out_scale", float(out_scale)),
            helper.make_attribute("out_bias", float(out_bias)),
            helper.make_attribute("out_dtype", output_dtype),
            helper.make_attribute("data_layout", "NHWC"),  # Input is in NHWC format
        ]
    )

    # Generate evenly-spaced threshold values within input dtype range
    # Use 80% of range to ensure values fit after rounding/clipping in FINN pipeline
    # Round to integers so FINN's MinimizeAccumulatorWidth can validate them
    inp_dt = DataType[input_dtype]
    inp_min, inp_max = inp_dt.min(), inp_dt.max()
    thresh_vals = np.linspace(inp_min * 0.8, inp_max * 0.8, num_thresholds, dtype=np.float32)
    thresh_vals = np.round(thresh_vals).astype(np.float32)  # Round to integers
    thresh_vals = np.tile(thresh_vals, (channels, 1))  # Replicate for each channel

    # Create graph
    graph = helper.make_graph(
        [node],
        "multithreshold_test",
        [inp],
        [outp],
        [
            helper.make_tensor(
                "thresh", TensorProto.FLOAT, thresh_shape, thresh_vals.flatten().tolist()
            )
        ],
    )

    model = ModelWrapper(qonnx_make_model(graph))

    # Set datatypes
    model.set_tensor_datatype("inp", DataType[input_dtype])
    model.set_tensor_datatype("thresh", DataType[threshold_dtype])
    model.set_tensor_datatype("outp", DataType[output_dtype])

    return model, node


def make_funclayernorm_model(
    shape: list[int] = [1, 128, 768],
    datatype: str = "INT8",
    epsilon: float = 1e-5,
    axis: int = -1,
    input_name: str = "inp",
    output_name: str = "out",
) -> tuple[ModelWrapper, NodeProto]:
    """Create FuncLayerNorm operation model for BERT-style normalization.

    Use this for LayerNorm operations with single input/output and shape preservation.
    Automatically applies InferShapes() and InferDataTypes() transforms.

    Args:
        shape: Input/output tensor shape (e.g., [1, 128, 768] for BERT-base)
        datatype: Tensor datatype (e.g., "INT8", "FLOAT32")
        epsilon: LayerNorm epsilon value for numerical stability (default: 1e-5)
        axis: Normalization axis (default: -1 for channel/last axis)
        input_name: Input tensor name (default: "inp")
        output_name: Output tensor name (default: "out")

    Returns:
        (model, node): ModelWrapper and FuncLayerNorm node

    Example:
        >>> # BERT-base LayerNorm (768 dimensions)
        >>> model, node = make_funclayernorm_model(
        ...     shape=[1, 128, 768],
        ...     datatype="INT8",
        ...     epsilon=1e-5
        ... )

        >>> # BERT-large LayerNorm (1024 dimensions)
        >>> model, node = make_funclayernorm_model(
        ...     shape=[1, 128, 1024],
        ...     datatype="INT16"
        ... )
    """
    inp = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)

    node = helper.make_node(
        "FuncLayerNorm", [input_name], [output_name], axis=axis, epsilon=epsilon
    )

    graph = helper.make_graph([node], "funclayernorm_test", [inp], [out])
    model = ModelWrapper(qonnx_make_model(graph))

    # Set datatypes
    dt = DataType[datatype]
    model.set_tensor_datatype(input_name, dt)
    model.set_tensor_datatype(output_name, dt)

    # Apply transforms
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model, node


def make_vvau_model(
    channels: int = 16,
    kernel_shape: list[int] = [3, 3],
    dim: list[int] = [28, 28],
    input_dtype: str = "INT8",
    weight_dtype: str = "INT8",
    output_dtype: str = "INT8",
    PE: int = 4,
    SIMD: int = 3,
    no_activation: int = 1,
    mode: str = "vvau_node",
) -> tuple[ModelWrapper, NodeProto]:
    """Create VVAU (Vector-Vector Activation) model with depthwise sparse weights.

    Use this for depthwise convolution tests. Supports two modes:
    - "vvau_node": Direct VectorVectorActivation node (for shape/backend tests)
    - "matmul_with_sparsity": MatMul + sparsity annotation (for inference tests)

    Args:
        channels: Number of output channels (e.g., 16)
        kernel_shape: Kernel size [k_h, k_w] (e.g., [3, 3])
        dim: Spatial dimensions [dim_h, dim_w] (e.g., [28, 28])
        input_dtype: Input datatype (e.g., "INT8")
        weight_dtype: Weight datatype (e.g., "INT8")
        output_dtype: Output datatype (e.g., "INT8")
        PE: Parallelization factor (default: 4)
        SIMD: SIMD width (default: 3)
        no_activation: Whether to skip activation (default: 1)
        mode: "vvau_node" or "matmul_with_sparsity" (default: "vvau_node")

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> # Direct VVAU node (for parity tests)
        >>> model, node = make_vvau_model(
        ...     channels=16,
        ...     kernel_shape=[3, 3],
        ...     dim=[28, 28],
        ...     mode="vvau_node"
        ... )

        >>> # MatMul with sparsity (for inference tests)
        >>> model, node = make_vvau_model(
        ...     channels=16,
        ...     kernel_shape=[3, 3],
        ...     mode="matmul_with_sparsity"
        ... )
    """
    k_h, k_w = kernel_shape
    dim_h, dim_w = dim

    # Input shape after Im2Col: [1, dim_h, dim_w, k_h * k_w * channels]
    # Output shape: [1, dim_h, dim_w, channels]
    input_shape = [1, dim_h, dim_w, k_h * k_w * channels]
    output_shape = [1, dim_h, dim_w, channels]

    # Weight shape for depthwise: [k_h * k_w * channels, channels]
    weight_shape = [k_h * k_w * channels, channels]

    # Create depthwise sparse weight matrix (diagonal blocks)
    W_sparse = np.zeros(weight_shape, dtype=np.float32)
    for ch in range(channels):
        # Fill diagonal elements (depthwise structure)
        for i in range(k_h * k_w):
            W_sparse[ch * k_h * k_w + i, ch] = np.random.rand()

    # Create tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

    if mode == "matmul_with_sparsity":
        # Create MatMul node for inference testing
        node = helper.make_node(
            "MatMul", inputs=["input", "weights"], outputs=["output"], name="MatMul_0"
        )

        weight_init = helper.make_tensor(
            "weights", TensorProto.FLOAT, weight_shape, W_sparse.flatten().tolist()
        )

        graph = helper.make_graph(
            [node], "vvau_matmul_test", [input_tensor], [output_tensor], [weight_init]
        )

        model = ModelWrapper(qonnx_make_model(graph))

        # Set datatypes
        model.set_tensor_datatype("input", DataType[input_dtype])
        model.set_tensor_datatype("weights", DataType[weight_dtype])
        model.set_tensor_datatype("output", DataType[output_dtype])

        # CRITICAL: Set sparsity annotation for inference
        model.set_tensor_sparsity("weights", {"dw": {"kernel_shape": kernel_shape}})

    else:  # mode == "vvau_node"
        # Create direct VectorVectorActivation node
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=PE,
            SIMD=SIMD,
            Dim=[dim_h, dim_w],
            Channels=channels,
            Kernel=kernel_shape,
            input0Datatype=input_dtype,
            input1Datatype=weight_dtype,
            output0Datatype=output_dtype,
            no_activation=no_activation,
        )

        weight_init = helper.make_tensor(
            "weights", TensorProto.FLOAT, weight_shape, W_sparse.flatten().tolist()
        )

        graph = helper.make_graph(
            [node], "vvau_test", [input_tensor], [output_tensor], [weight_init]
        )

        model = ModelWrapper(qonnx_make_model(graph))

        # Set datatypes
        model.set_tensor_datatype("input", DataType[input_dtype])
        model.set_tensor_datatype("weights", DataType[weight_dtype])
        model.set_tensor_datatype("output", DataType[output_dtype])

    return model, node


def make_broadcast_model(
    lhs_shape: list[int],
    rhs_shape: list[int],
    operation: str = "Add",
    datatype: str = "INT8",
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
    output_name: str = "output",
) -> tuple[ModelWrapper, NodeProto]:
    """Create binary operation model with broadcasting semantics.

    Use this for testing broadcasting behavior in elementwise operations.
    Automatically computes correct output shape using numpy broadcasting rules.

    Args:
        lhs_shape: Left-hand side input shape (e.g., [1, 64, 64, 128])
        rhs_shape: Right-hand side input shape (e.g., [128] or [1, 1, 1, 128])
        operation: Binary operation (e.g., "Add", "Sub", "Mul", "Div")
        datatype: Tensor datatype (default: "INT8")
        lhs_name: Left input name (default: "lhs")
        rhs_name: Right input name (default: "rhs")
        output_name: Output name (default: "output")

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> # Channel broadcasting: [1,64,64,128] + [128] → [1,64,64,128]
        >>> model, node = make_broadcast_model([1,64,64,128], [128], "Add")

        >>> # Scalar broadcasting: [1,64,64,128] + [1] → [1,64,64,128]
        >>> model, node = make_broadcast_model([1,64,64,128], [1], "Mul")

        >>> # Spatial broadcasting: [1,64,64,128] + [1,1,1,128]
        >>> model, node = make_broadcast_model(
        ...     [1,64,64,128],
        ...     [1,1,1,128],
        ...     "Add"
        ... )

        >>> # Bidirectional broadcasting: [1,64,1,128] + [1,1,64,1] → [1,64,64,128]
        >>> model, node = make_broadcast_model([1,64,1,128], [1,1,64,1], "Add")
    """
    # Compute output shape using numpy broadcasting rules
    output_shape = tuple(np.broadcast_shapes(lhs_shape, rhs_shape))

    # Create input tensor infos
    lhs_input = helper.make_tensor_value_info(lhs_name, TensorProto.FLOAT, lhs_shape)
    rhs_input = helper.make_tensor_value_info(rhs_name, TensorProto.FLOAT, rhs_shape)
    output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)

    # Create ONNX node
    node = helper.make_node(
        operation, [lhs_name, rhs_name], [output_name], name=f"{operation}_broadcast_test"
    )

    # Build graph and model
    graph = helper.make_graph(
        nodes=[node],
        name="broadcast_test",
        inputs=[lhs_input, rhs_input],
        outputs=[output],
    )
    model = ModelWrapper(qonnx_make_model(graph))

    # Set datatypes
    dt = DataType[datatype]
    model.set_tensor_datatype(lhs_name, dt)
    model.set_tensor_datatype(rhs_name, dt)
    model.set_tensor_datatype(output_name, dt)

    return model, node


def make_duplicate_streams_model(
    shape: list[int] = [1, 8, 8, 64],
    num_outputs: int = 2,
    datatype: str = "INT8",
    PE: int | None = None,
    input_name: str = "inp",
    output_prefix: str = "out",
    mode: str = "duplicate_node",
) -> tuple[ModelWrapper, NodeProto]:
    """Create DuplicateStreams model with variable-arity outputs.

    Use this for testing stream fanout/duplication. Supports two modes:
    - "duplicate_node": Direct DuplicateStreams node (for parity/execution tests)
    - "fanout_graph": Multi-node graph with natural fanout (for transform tests)

    Args:
        shape: Input tensor shape (e.g., [1, 8, 8, 64])
        num_outputs: Number of duplicate output streams (default: 2)
        datatype: Tensor datatype (default: "INT8")
        PE: Parallelization factor. If None, auto-computes as shape[-1]
        input_name: Input tensor name (default: "inp")
        output_prefix: Prefix for output names (default: "out")
                      Outputs will be named "out0", "out1", ..., "out{N-1}"
        mode: "duplicate_node" or "fanout_graph" (default: "duplicate_node")

    Returns:
        (model, node): ModelWrapper and operation node

    Example:
        >>> # Direct DuplicateStreams node (parity tests)
        >>> model, node = make_duplicate_streams_model(
        ...     shape=[1, 8, 8, 64],
        ...     num_outputs=2,
        ...     mode="duplicate_node"
        ... )

        >>> # 4-way duplication
        >>> model, node = make_duplicate_streams_model(
        ...     shape=[1, 8, 8, 64],
        ...     num_outputs=4
        ... )

        >>> # Fanout graph (transform tests)
        >>> model, node = make_duplicate_streams_model(
        ...     shape=[1, 8, 8, 64],
        ...     num_outputs=3,
        ...     mode="fanout_graph"
        ... )
    """
    channels = shape[-1]
    PE = PE if PE is not None else channels

    # Generate output names
    output_names = [f"{output_prefix}{i}" for i in range(num_outputs)]

    # Create input tensor info
    inp = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)

    # Create output tensor infos (all same shape as input)
    outputs_info = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in output_names
    ]

    if mode == "fanout_graph":
        # Create multi-node graph with natural fanout
        # Conv → tensor_x → [Add, Mul, ...]
        weight_shape = [channels, channels, 3, 3]  # Conv weights
        weight_init = helper.make_tensor(
            "weight",
            TensorProto.FLOAT,
            weight_shape,
            np.random.rand(*weight_shape).flatten().tolist(),
        )

        # Intermediate tensor that fans out
        helper.make_tensor_value_info("tensor_x", TensorProto.FLOAT, shape)

        # Create Conv node (producer)
        conv = helper.make_node("Conv", [input_name, "weight"], ["tensor_x"], name="Conv_0")

        # Create consumer nodes (one per output)
        consumers = []
        for i, out_name in enumerate(output_names):
            if i == 0:
                op = "Add"
                param_name = "bias"
            elif i == 1:
                op = "Mul"
                param_name = "scale"
            else:
                op = "Add" if i % 2 == 0 else "Mul"
                param_name = f"param{i}"

            # Create parameter
            param_init = helper.make_tensor(
                param_name, TensorProto.FLOAT, [channels], np.ones(channels).flatten().tolist()
            )

            node = helper.make_node(op, ["tensor_x", param_name], [out_name], name=f"{op}_{i}")
            consumers.append((node, param_init))

        all_nodes = [conv] + [c[0] for c in consumers]
        all_inits = [weight_init] + [c[1] for c in consumers]

        graph = helper.make_graph(all_nodes, "fanout_graph", [inp], outputs_info, all_inits)

        model = ModelWrapper(qonnx_make_model(graph))

        # Set datatypes
        dt = DataType[datatype]
        model.set_tensor_datatype(input_name, dt)
        model.set_tensor_datatype("tensor_x", dt)
        for out_name in output_names:
            model.set_tensor_datatype(out_name, dt)

        # Return the Conv node (the one being transformed)
        return model, conv

    else:  # mode == "duplicate_node"
        # Create direct DuplicateStreams node
        node = helper.make_node(
            "DuplicateStreams",
            inputs=[input_name],
            outputs=output_names,
            domain="finn.custom_op.fpgadataflow",
            NumChannels=channels,
            PE=PE,
            NumOutputStreams=num_outputs,
            inputDataType=datatype,
            numInputVectors=shape[1:-1],  # Spatial dimensions
        )

        graph = helper.make_graph([node], "duplicate_streams_test", [inp], outputs_info)
        model = ModelWrapper(qonnx_make_model(graph))

        # Set datatypes
        dt = DataType[datatype]
        model.set_tensor_datatype(input_name, dt)
        for out_name in output_names:
            model.set_tensor_datatype(out_name, dt)

        return model, node
