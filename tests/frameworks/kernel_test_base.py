"""Abstract base class for kernel tests with compositional configuration.

Provides shared utilities for KernelTest and KernelParityTest:
- Model preparation with DataType annotations
- Test data generation
- Golden reference validation
- Backend specialization (Stage 2 → Stage 3)
- Declarative configuration via fixtures

Subclasses must implement:
- make_test_model(): Create ONNX model to test
- get_kernel_op(): Return kernel operator class

Config passed as parameter (composition), not inherited.
"""

from abc import ABC, abstractmethod

import numpy as np
import pytest
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import getHWCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from brainsmith.primitives.transforms.infer_kernels import InferKernels
from brainsmith.registry import get_backend, list_backends_for_kernel
from tests.fixtures.model_annotation import annotate_model_datatypes
from tests.fixtures.test_data import generate_test_data
from tests.frameworks.test_config import KernelTestConfig
from tests.support.backend_utils import specialize_to_backend
from tests.support.executors import CppSimExecutor, PythonExecutor, RTLSimExecutor
from tests.support.validator import GoldenValidator


class KernelTestBase(ABC):
    """Abstract base class for kernel tests with compositional configuration.

    Subclasses: KernelTest, KernelParityTest

    Abstract methods (must implement):
    - make_test_model(): Create ONNX model to test
    - get_kernel_op(): Return kernel operator class

    Shared utilities:
    - validate_against_golden(): Output validation
    - specialize_to_backend(): Stage 2→3 specialization
    - auto_configure_from_fixture(): Apply DSE parameters
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement
    # ========================================================================

    @abstractmethod
    def make_test_model(
        self, kernel_test_config: KernelTestConfig
    ) -> tuple[ModelWrapper, list[str]]:
        """Create ONNX model to test (framework adds DataType annotations).

        Args:
            kernel_test_config: Test configuration with shapes, dtypes, operation params

        Returns:
            (model, input_names): ONNX model and list of input tensor names
        """
        pass

    # ========================================================================
    # Optional Configuration Hooks
    # ========================================================================

    def get_test_seed(self) -> int:
        """Return random seed for test data generation (optional).

        Returns:
            Random seed integer (default: 42)
        """
        return 42

    def get_backend_variants(self) -> list[type] | None:
        """Backend variant classes for specialization in priority order.

        Returns:
            List of backend classes to try in priority order.
            Default: None (auto-detect HLS backend based on kernel op_type)
        """
        return None

    # ========================================================================
    # Stage 2: Kernel Inference
    # ========================================================================

    @abstractmethod
    def get_kernel_op(self) -> type:
        """Return kernel operator class.

        Returns:
            Kernel operator class (e.g., ElementwiseBinaryOp, AddStreams)
        """
        pass

    def infer_kernel(
        self,
        model: ModelWrapper,
        target_node: str,
    ) -> tuple[HWCustomOp, ModelWrapper]:
        """Execute Stage 1 → Stage 2 kernel inference.

        Default: uses InferKernels([get_kernel_op()])
        Override: for custom inference logic (multiple transforms, conditional, etc.)

        Args:
            model: Stage 1 model (ONNX nodes)
            target_node: Name of target ONNX node to transform

        Returns:
            (op, model): Kernel operator instance and transformed model
        """
        # Create transform with kernel op from subclass
        transform = InferKernels([self.get_kernel_op()])

        # Apply inference transform
        model = model.transform(transform)

        # Find transformed node
        op = self._find_hw_node(model, target_node)

        return op, model

    # ========================================================================
    # Shared Helper Methods (Extracted from KernelTest/KernelParityTest)
    # ========================================================================

    def _prepare_model_with_annotations(
        self, kernel_test_config: "KernelTestConfig"
    ) -> tuple[ModelWrapper, str]:
        """Create model with QONNX DataType annotations (internal helper).

        Args:
            kernel_test_config: Test configuration with shapes, dtypes

        Returns:
            (model, target_node): Model with DataType annotations and target node name
        """
        # Step 1: Get simple model from test
        model, input_names = self.make_test_model(kernel_test_config)

        # Step 2: Annotate inputs with QONNX DataTypes (direct annotation, no Quant nodes)
        input_datatypes = kernel_test_config.input_dtypes
        input_annotations = {
            name: input_datatypes[name] for name in input_names if name in input_datatypes
        }
        if input_annotations:
            annotate_model_datatypes(model, input_annotations, warn_unsupported=True)

        # Step 3: Annotate outputs (infer from model or use defaults)
        output_names = [out.name for out in model.graph.output]
        for out_name in output_names:
            if model.get_tensor_datatype(out_name) is None:
                # Default: same dtype as first input
                first_input = input_names[0]
                if first_input in input_datatypes:
                    model.set_tensor_datatype(out_name, input_datatypes[first_input])

        # Step 4: Find target node (first node in graph)
        if len(model.graph.node) == 0:
            raise RuntimeError("No nodes found in graph")

        target_node = model.graph.node[0].name

        return model, target_node

    def _generate_test_inputs(
        self, kernel_test_config: "KernelTestConfig"
    ) -> dict[str, np.ndarray]:
        """Generate test data with correct shapes and datatypes (internal helper).

        Args:
            kernel_test_config: Test configuration

        Returns:
            Dict mapping input names to test data arrays
        """
        inputs = {}
        input_shapes = kernel_test_config.input_shapes
        input_datatypes = kernel_test_config.input_dtypes

        # Input names come from config keys (no need to create model)
        for name in input_datatypes.keys():
            dtype = input_datatypes[name]
            shape = input_shapes[name]
            inputs[name] = generate_test_data(dtype, shape, seed=self.get_test_seed())

        return inputs

    def _find_hw_node(
        self,
        model: ModelWrapper,
        target_node: str,
        expected_type=None,
    ) -> HWCustomOp:
        """Find and validate HW node after kernel inference.

        Args:
            model: Model after inference transform
            target_node: Original ONNX node name (may have been renamed during transformation)
            expected_type: Expected kernel class or op_type string (optional)

        Returns:
            Kernel operator instance

        Raises:
            AssertionError: If node not found or wrong type
        """
        # Try to get ONNX node by name (may fail with new sequential naming)
        onnx_node = model.get_node_from_name(target_node)

        # If not found by exact name, search by kernel type
        # (needed for new sequential naming scheme: KernelName_<index>)
        if onnx_node is None:
            kernel_op = self.get_kernel_op()
            kernel_name = kernel_op.__name__  # e.g., "Thresholding"

            # Find all nodes with matching op_type and brainsmith domain
            hw_nodes = [
                node for node in model.graph.node
                if node.op_type == kernel_name and node.domain == "brainsmith.kernels"
            ]

            # For test models with single kernel instance, take the first one
            assert len(hw_nodes) > 0, (
                f"Could not find transformed kernel node. "
                f"Original node: {target_node}, Expected kernel: {kernel_name}"
            )
            assert len(hw_nodes) == 1, (
                f"Found multiple {kernel_name} nodes: {[n.name for n in hw_nodes]}. "
                f"Cannot determine which corresponds to {target_node}"
            )
            onnx_node = hw_nodes[0]

        # Wrap with custom op class (pass model for KernelOp initialization)
        op = getHWCustomOp(onnx_node, model)

        # Verify it's a HW node
        assert isinstance(
            op, HWCustomOp
        ), f"Node {target_node} is not a hardware operator (found {type(op).__name__})"

        # Type checking if expected_type provided
        if expected_type is not None:
            if isinstance(expected_type, type):
                # Class check
                assert isinstance(
                    op, expected_type
                ), f"Node {target_node} is {type(op).__name__}, expected {expected_type.__name__}"
            elif isinstance(expected_type, str):
                # op_type string check
                assert (
                    op.onnx_node.op_type == expected_type
                ), f"Node {target_node} has op_type {op.onnx_node.op_type}, expected {expected_type}"

        return op

    def _compute_golden_reference(
        self,
        quant_model: ModelWrapper,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute golden reference using QONNX execution on Stage 1 model.

        Args:
            quant_model: ModelWrapper WITH DataType annotations
            inputs: Test data (pre-quantized)

        Returns:
            Expected outputs from QONNX execution
        """
        from qonnx.core.datatype import DataType

        outputs = execute_onnx(quant_model, inputs, return_full_exec_context=False)

        # Post-process BIPOLAR outputs
        # QONNX MultiThreshold returns {0, 1} but BIPOLAR datatype represents {-1, 1}
        # Apply the same conversion as hardware Thresholding kernels
        graph = quant_model.graph
        for node in graph.node:
            for output_name in node.output:
                if output_name in outputs:
                    output_dtype = quant_model.get_tensor_datatype(output_name)
                    if output_dtype == DataType["BIPOLAR"]:
                        outputs[output_name] = 2 * outputs[output_name] - 1

        return outputs

    def _build_stage1_model(self, kernel_test_config: "KernelTestConfig") -> ModelWrapper:
        """Build Stage 1 model with QONNX annotations.

        Args:
            kernel_test_config: Test configuration

        Returns:
            Stage 1 model (ONNX + annotations, no kernel inference)
        """
        model, _ = self._prepare_model_with_annotations(kernel_test_config)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        return model

    def _build_test_inputs(self, kernel_test_config: "KernelTestConfig") -> dict[str, np.ndarray]:
        """Build test inputs with deterministic seed.

        Args:
            kernel_test_config: Test configuration

        Returns:
            Dict mapping input names to test data arrays
        """
        return self._generate_test_inputs(kernel_test_config)

    def _build_golden_outputs(
        self, stage1_model: ModelWrapper, test_inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Build golden outputs from Stage 1 model.

        Args:
            stage1_model: Stage 1 model
            test_inputs: Test inputs

        Returns:
            Expected outputs from QONNX execution
        """
        return self._compute_golden_reference(stage1_model, test_inputs)

    # ========================================================================
    # Shared Utility: Execute and Validate Against Golden Reference
    # ========================================================================

    def _execute_and_validate_golden(
        self,
        stage_model: tuple[HWCustomOp, ModelWrapper],
        test_inputs: dict[str, np.ndarray],
        golden_outputs: dict[str, np.ndarray],
        execution_mode: str,
        backend_name: str,
        config: KernelTestConfig,
    ) -> None:
        """Execute kernel and validate against golden reference.

        Args:
            stage_model: (op, model) tuple from fixture
            test_inputs: Test data
            golden_outputs: Expected outputs
            execution_mode: "python", "cppsim", or "rtlsim"
            backend_name: Backend name for error messages
            config: Test configuration with tolerances

        Raises:
            AssertionError: If outputs don't match golden within tolerance
            ValueError: If invalid execution_mode
        """
        op, model = stage_model

        # Select executor based on mode
        if execution_mode == "python":
            executor = PythonExecutor()
        elif execution_mode == "cppsim":
            executor = CppSimExecutor()
        elif execution_mode == "rtlsim":
            executor = RTLSimExecutor()
        else:
            raise ValueError(f"Invalid execution_mode: {execution_mode}")

        # Execute kernel
        actual_outputs = executor.execute(op, model, test_inputs)

        # Get tolerance based on execution mode
        if execution_mode == "python":
            tolerance = config.tolerance_python
        elif execution_mode == "cppsim":
            tolerance = config.tolerance_cppsim
        elif execution_mode == "rtlsim":
            tolerance = config.validation.get_tolerance_rtlsim()
        else:
            raise ValueError(f"Invalid execution_mode: {execution_mode}")

        # Validate against golden
        GoldenValidator().validate(
            actual_outputs=actual_outputs,
            golden_outputs=golden_outputs,
            backend_name=backend_name,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        )

    # ========================================================================
    # Shared Utility 2: Backend Auto-Detection
    # ========================================================================

    def _auto_detect_backends(self, op: HWCustomOp) -> list[type]:
        """Auto-detect backend variants from Brainsmith registry.

        Args:
            op: HWCustomOp instance to find backends for

        Returns:
            List of backend classes

        Raises:
            pytest.skip: If no backends found
        """
        backend_names = list_backends_for_kernel(op.onnx_node.op_type, language="hls")
        if not backend_names:
            pytest.skip(f"No HLS backend found for {op.onnx_node.op_type}")

        return [get_backend(name) for name in backend_names]

    # ========================================================================
    # Shared Utility 3: Backend Specialization (Stage 2 → Stage 3)
    # ========================================================================

    def specialize_to_backend(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        config: KernelTestConfig,
        backend_variants_override: list[type] | None = None,
    ) -> tuple[HWCustomOp, ModelWrapper]:
        """Execute Stage 2 → Stage 3 backend specialization.

        Default: auto-detects backends from registry
        Override: for custom specialization logic

        Args:
            op: Base kernel operator (Stage 2)
            model: Model containing base kernel
            config: Test configuration (contains fpgapart)
            backend_variants_override: Optional backend list (for parity testing)

        Returns:
            (specialized_op, specialized_model): Backend operator and model (Stage 3)
        """
        fpgapart = config.fpgapart
        if fpgapart is None:
            raise ValueError(
                "Backend specialization requires fpgapart to be configured. "
                "Set platform=PlatformConfig(fpgapart='...') in your kernel_test_config."
            )

        # Determine backend variants (priority: override > get_backend_variants > auto-detect)
        backend_variants = backend_variants_override
        if backend_variants is None:
            backend_variants = self.get_backend_variants()
        if backend_variants is None:
            backend_variants = self._auto_detect_backends(op)

        # Specialize to backend (Stage 2 → Stage 3)
        op, model = specialize_to_backend(op, model, fpgapart, backend_variants)

        return op, model

    # ========================================================================
    # Shared Utility 4: Stage-Aware Parameter Configuration
    # ========================================================================

    def get_stage(self, op) -> int:
        """Detect pipeline stage from backend attribute.

        Args:
            op: Operator instance (HWCustomOp or KernelOp)

        Returns:
            2: Stage 2 (kernel, unspecialized - backend="fpgadataflow")
            3: Stage 3 (backend, specialized to HLS or RTL)
        """
        backend = op.get_nodeattr("backend")
        return 3 if backend in ("hls", "rtl") else 2

    def is_brainsmith(self, op) -> bool:
        """Check if node is Brainsmith KernelOp.

        Args:
            op: Operator instance

        Returns:
            True if op has kernel_schema attribute (Brainsmith KernelOp)
            False if FINN HWCustomOp only
        """
        return hasattr(op, "kernel_schema")

    # ========================================================================
    # Shared Utility 5: Auto-Configuration from Fixture
    # ========================================================================

    def auto_configure_from_fixture(
        self, op, model: ModelWrapper, stage: int, config: KernelTestConfig
    ) -> None:
        """Auto-apply DSE parameters from test configuration.

        Applies input_streams, output_streams, and dse_dimensions from config.

        Args:
            op: Operator instance (KernelOp or HWCustomOp)
            model: ModelWrapper
            stage: Pipeline stage (2=kernel, 3=backend)
            config: KernelTestConfig with declarative parameters
        """
        # FINN nodes: use direct nodeattr setting
        if not self.is_brainsmith(op):
            self._apply_finn_config(op, config, stage)
            return

        # Brainsmith nodes: use semantic DSE API
        op._ensure_ready(model)
        point = op.design_point

        # Apply stream dimensions (semantic API) - available at Stage 2+
        if stage >= 2:
            # Input stream parallelism (e.g., PE)
            if config.input_streams:
                for idx, value in config.input_streams.items():
                    point = point.with_input_stream(idx, value)

            # Output stream parallelism
            if config.output_streams:
                for idx, value in config.output_streams.items():
                    point = point.with_output_stream(idx, value)

        # Apply other DSE dimensions (generic API)
        if config.dse_dimensions:
            for name, value in config.dse_dimensions.items():
                point = point.with_dimension(name, value)

        # Apply the configured design point
        op.apply_design_point(point)

    def _apply_finn_config(self, op, config: KernelTestConfig, stage: int):
        """Apply configuration to FINN HWCustomOp node.

        FINN nodes use direct nodeattr mutation instead of design points.

        Args:
            op: FINN HWCustomOp instance
            config: KernelTestConfig
            stage: Pipeline stage
        """
        # Stream dimensions - FINN typically uses PE/SIMD directly
        if stage >= 2:
            if config.input_streams:
                # FINN convention: first input stream → PE
                if 0 in config.input_streams:
                    op.set_nodeattr("PE", config.input_streams[0])

            if config.output_streams:
                # FINN convention: first output stream → output PE
                if 0 in config.output_streams:
                    # Some FINN nodes have separate output parallelism
                    try:
                        op.set_nodeattr("output_PE", config.output_streams[0])
                    except (AttributeError, KeyError):
                        pass  # Not all FINN nodes have output_PE

        # Other DSE dimensions
        if config.dse_dimensions:
            for name, value in config.dse_dimensions.items():
                try:
                    op.set_nodeattr(name, value)
                except (AttributeError, KeyError):
                    # Skip params that don't exist on this node
                    pass
