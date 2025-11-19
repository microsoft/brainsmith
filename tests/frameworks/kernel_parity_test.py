"""Test framework for comparing two kernel implementations (parity testing).

Compares reference vs primary implementations (e.g., FINN vs Brainsmith).

Provides 18 inherited tests:
- 7 core parity tests (shapes, datatypes, stream widths)
- 5 HW estimation parity tests (cycles, resources, efficiency)
- 6 golden execution tests (reference/primary × Python/cppsim/rtlsim)

Subclasses must implement:
- make_test_model(): Create shared ONNX model
- infer_kernel_reference(): Reference kernel inference
- get_backend_variants_reference(): Reference backends
- get_num_inputs(), get_num_outputs(): Validation counts

Primary implementation uses inherited defaults from KernelTestBase.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pytest
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.modelwrapper import ModelWrapper

from tests.frameworks.kernel_test_base import KernelTestBase
from tests.support.assertions import (
    assert_datatypes_match,
    assert_shapes_match,
    assert_values_match,
    assert_widths_match,
)
from tests.support.backend_utils import specialize_to_backend

if TYPE_CHECKING:
    from tests.frameworks.test_config import KernelTestConfig


class KernelParityTest(KernelTestBase):
    """Compare two kernel implementations for parity.

    Abstract methods (must implement):
    - make_test_model(): Create shared ONNX model
    - infer_kernel_reference(): Reference kernel inference
    - get_backend_variants_reference(): Reference backends
    - get_num_inputs(), get_num_outputs(): Validation counts

    Test suite: 18 tests (7 parity + 5 HW estimation + 6 golden execution)
    """

    # ========================================================================
    # Abstract Methods - Subclasses MUST implement (4 total)
    # ========================================================================

    @abstractmethod
    def get_num_inputs(self) -> int:
        """Return number of inputs for validation.

        Used by parity tests to validate shapes/datatypes have correct count.

        Returns:
            Number of inputs to kernel

        Example:
            def get_num_inputs(self):
                return 2  # Binary operation has 2 inputs
        """
        pass

    @abstractmethod
    def get_num_outputs(self) -> int:
        """Return number of outputs for validation.

        Used by parity tests to validate shapes/datatypes have correct count.

        Returns:
            Number of outputs from kernel

        Example:
            def get_num_outputs(self):
                return 1  # Most operations have 1 output
        """
        pass

    # ========================================================================
    # Reference-Based API - Asymmetric Design
    # ========================================================================

    @abstractmethod
    def infer_kernel_reference(
        self,
        model: ModelWrapper,
        target_node: str,
    ) -> tuple[HWCustomOp, ModelWrapper]:
        """Infer reference implementation.

        Reference implementation uses this method to transform ONNX → Reference Kernel.
        Primary implementation uses inherited infer_kernel() from base class.

        Args:
            model: Stage 1 model (ONNX with annotations)
            target_node: Target node name

        Returns:
            (op, model): Reference kernel and model

        Example:
            def infer_kernel_reference(self, model, target_node):
                # FINN inference
                from finn.transformation.fpgadataflow.convert_to_hw_layers import (
                    InferElementwiseBinaryOperation,
                )
                model = model.transform(InferElementwiseBinaryOperation())
                nodes = model.get_nodes_by_op_type("ElementwiseAdd")
                from qonnx.custom_op.registry import getCustomOp
                return getCustomOp(nodes[0]), model
        """
        pass

    @abstractmethod
    def get_backend_variants_reference(self) -> list[type]:
        """Return backend variants for reference implementation.

        Reference implementation uses this method to specify HLS/RTL backends.
        Primary implementation uses inherited get_backend_variants() from base class.

        Returns:
            List of backend classes

        Example:
            def get_backend_variants_reference(self):
                from finn.custom_op.fpgadataflow.hls.elementwise_binary_hls import (
                    ElementwiseAdd_hls,
                )
                return [ElementwiseAdd_hls]
        """
        pass

    def configure_kernel_reference(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        stage: int,
        config: "KernelTestConfig",
    ):
        """Configure reference kernel parameters.

        Reference implementation uses this method for configuration.
        Primary implementation uses inherited configure_kernel() from base class.

        Default: Uses auto_configure_from_fixture() - override if custom logic needed.

        Args:
            op: Reference operator
            model: Model
            stage: Pipeline stage (2=kernel, 3=backend)
            config: Test configuration
        """
        self.auto_configure_from_fixture(op, model, stage, config)

    def specialize_to_backend_reference(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        config: "KernelTestConfig",
    ) -> tuple[HWCustomOp, ModelWrapper]:
        """Specialize reference to backend.

        Uses explicit backend variants (no method swapping).

        Default: Uses get_backend_variants_reference() and shared logic.
        Override: Custom specialization for reference implementation.

        Args:
            op: Reference operator (Stage 2)
            model: Model
            config: Configuration

        Returns:
            (backend_op, model): Specialized reference and model
        """
        fpgapart = config.fpgapart
        if fpgapart is None:
            pytest.skip("Backend testing skipped (no FPGA part configured)")

        backend_variants = self.get_backend_variants_reference()

        return specialize_to_backend(op, model, fpgapart, backend_variants)

    @pytest.fixture(scope="function")
    def stage1_model(self, kernel_test_config: "KernelTestConfig", model_cache) -> ModelWrapper:
        """Stage 1 model with QONNX annotations (shared by both kernels).

        Same as KernelTest.stage1_model - creates ONNX model before kernel inference.

        Args:
            kernel_test_config: Unified test configuration
            model_cache: Session-scoped cache for model artifacts

        Returns:
            Stage 1 model (ONNX + annotations, no kernel inference)
        """

        def builder():
            return self._build_stage1_model(kernel_test_config)

        return model_cache.get_stage1_model(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def test_inputs(
        self, kernel_test_config: "KernelTestConfig", model_cache
    ) -> dict[str, np.ndarray]:
        """Generate test inputs (shared by both kernels).

        Same as KernelTest.test_inputs - generates test data.

        Args:
            kernel_test_config: Unified test configuration
            model_cache: Session-scoped cache for model artifacts

        Returns:
            Dict mapping input names to test data arrays (pre-quantized)
        """

        def builder():
            return self._build_test_inputs(kernel_test_config)

        return model_cache.get_test_inputs(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def golden_outputs(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        test_inputs: dict[str, np.ndarray],
        model_cache,
    ) -> dict[str, np.ndarray]:
        """Golden reference from Stage 1 ONNX (shared by both kernels).

        Same as KernelTest.golden_outputs - computes golden reference once.

        Args:
            kernel_test_config: Unified test configuration
            stage1_model: Stage 1 model fixture
            test_inputs: Test inputs fixture
            model_cache: Session-scoped cache

        Returns:
            Expected outputs from QONNX execution
        """

        def builder():
            return self._build_golden_outputs(stage1_model, test_inputs)

        return model_cache.get_golden_outputs(kernel_test_config.test_id, builder)

    # ========================================================================
    # Pytest Fixtures
    # ========================================================================

    @pytest.fixture(scope="function")
    def stage2_model(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        model_cache,
    ) -> tuple:
        """Stage 2 primary model.

        Uses inherited infer_kernel() from base class.

        Returns (op, model): Primary kernel and model.
        """

        def builder():
            model = stage1_model
            target_node = model.graph.node[0].name

            # Use inherited infer_kernel() from base class
            op, model = self.infer_kernel(model, target_node)

            # Apply declarative configuration from fixture
            self.auto_configure_from_fixture(op, model, stage=2, config=kernel_test_config)

            return op, model

        return model_cache.get_stage2_model(kernel_test_config.test_id, builder)

    @pytest.fixture(scope="function")
    def stage2_model_reference(
        self,
        kernel_test_config: "KernelTestConfig",
        stage1_model: ModelWrapper,
        model_cache,
    ) -> tuple:
        """Stage 2 reference model.

        Uses infer_kernel_reference() method.

        Returns (op, model): Reference kernel and model.
        """

        def builder():
            model = stage1_model
            target_node = model.graph.node[0].name

            op, model = self.infer_kernel_reference(model, target_node)
            self.configure_kernel_reference(op, model, stage=2, config=kernel_test_config)

            return op, model

        cache_key = f"{kernel_test_config.test_id}_reference"
        return model_cache.get_stage2_model(cache_key, builder)

    @pytest.fixture(scope="function")
    def stage3_model(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model: tuple,
        model_cache,
    ) -> tuple:
        """Stage 3 primary model.

        Uses inherited specialize_to_backend() from base class.

        Returns (op, model): Primary backend and model.
        """
        fpgapart = kernel_test_config.fpgapart

        def builder():
            base_op, base_model = stage2_model
            op, model = self.specialize_to_backend(base_op, base_model, kernel_test_config)

            # Configure backend
            self.auto_configure_from_fixture(op, model, stage=3, config=kernel_test_config)

            return op, model

        return model_cache.get_stage3_model(kernel_test_config.test_id, fpgapart, builder)

    @pytest.fixture(scope="function")
    def stage3_model_reference(
        self,
        kernel_test_config: "KernelTestConfig",
        stage2_model_reference: tuple,
        model_cache,
    ) -> tuple:
        """Stage 3 reference model.

        Uses specialize_to_backend_reference() method.

        Returns (op, model): Reference backend and model.
        """
        fpgapart = kernel_test_config.fpgapart

        def builder():
            base_op, base_model = stage2_model_reference
            op, model = self.specialize_to_backend_reference(
                base_op, base_model, kernel_test_config
            )
            self.configure_kernel_reference(op, model, stage=3, config=kernel_test_config)
            return op, model

        cache_key = f"{kernel_test_config.test_id}_reference"
        return model_cache.get_stage3_model(cache_key, fpgapart, builder)

    @pytest.mark.golden
    @pytest.mark.kernel_parity
    def test_python_vs_golden(self, kernel_test_config, stage2_model, test_inputs, golden_outputs):
        """Test primary Python execution matches golden reference."""
        self._execute_and_validate_golden(
            stage2_model,
            test_inputs,
            golden_outputs,
            "python",
            "Primary Python execution",
            kernel_test_config,
        )

    @pytest.mark.golden
    @pytest.mark.kernel_parity
    def test_reference_python_vs_golden(
        self, kernel_test_config, stage2_model_reference, test_inputs, golden_outputs
    ):
        """Test reference Python execution matches golden reference."""
        self._execute_and_validate_golden(
            stage2_model_reference,
            test_inputs,
            golden_outputs,
            "python",
            "Reference Python execution",
            kernel_test_config,
        )

    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.kernel_parity
    def test_cppsim_vs_golden(self, kernel_test_config, stage3_model, test_inputs, golden_outputs):
        """Test primary cppsim execution matches golden reference."""
        self._execute_and_validate_golden(
            stage3_model,
            test_inputs,
            golden_outputs,
            "cppsim",
            "Primary HLS cppsim",
            kernel_test_config,
        )

    @pytest.mark.golden
    @pytest.mark.cppsim
    @pytest.mark.slow
    @pytest.mark.kernel_parity
    def test_reference_cppsim_vs_golden(
        self, kernel_test_config, stage3_model_reference, test_inputs, golden_outputs
    ):
        """Test reference cppsim execution matches golden reference."""
        self._execute_and_validate_golden(
            stage3_model_reference,
            test_inputs,
            golden_outputs,
            "cppsim",
            "Reference HLS cppsim",
            kernel_test_config,
        )

    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.kernel_parity
    def test_rtlsim_vs_golden(self, kernel_test_config, stage3_model, test_inputs, golden_outputs):
        """Test primary rtlsim execution matches golden reference."""
        self._execute_and_validate_golden(
            stage3_model,
            test_inputs,
            golden_outputs,
            "rtlsim",
            "Primary RTL rtlsim",
            kernel_test_config,
        )

    @pytest.mark.golden
    @pytest.mark.rtlsim
    @pytest.mark.slow
    @pytest.mark.kernel_parity
    def test_reference_rtlsim_vs_golden(
        self, kernel_test_config, stage3_model_reference, test_inputs, golden_outputs
    ):
        """Test reference rtlsim execution matches golden reference."""
        self._execute_and_validate_golden(
            stage3_model_reference,
            test_inputs,
            golden_outputs,
            "rtlsim",
            "Reference RTL rtlsim",
            kernel_test_config,
        )

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_normal_shapes_parity(self, kernel_test_config, stage2_model, stage2_model_reference):
        """Test normal input/output shapes match between implementations."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        # Input shapes
        for i in range(self.get_num_inputs()):
            shape = op.get_normal_input_shape(i)
            shape_ref = op_ref.get_normal_input_shape(i)

            # FINN Bug: get_normal_input_shape(ind) ignores index for multi-input nodes
            # FINN's Thresholding.get_normal_input_shape() always returns activation shape,
            # even for threshold input (ind=1). Skip comparison for non-zero indices.
            if i > 0 and self.get_num_inputs() > 1:
                pytest.skip(
                    f"FINN limitation: get_normal_input_shape({i}) incorrectly returns "
                    f"activation shape {shape_ref} instead of parameter shape {shape}. "
                    f"FINN's implementation ignores the 'ind' parameter for multi-input nodes."
                )

            assert_shapes_match(shape, shape_ref, i, "normal input")

        # Output shapes
        for i in range(self.get_num_outputs()):
            shape = op.get_normal_output_shape(i)
            shape_ref = op_ref.get_normal_output_shape(i)
            assert_shapes_match(shape, shape_ref, i, "normal output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_folded_shapes_parity(self, kernel_test_config, stage2_model, stage2_model_reference):
        """Test folded input/output shapes match between implementations."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        # Input shapes
        for i in range(self.get_num_inputs()):
            shape = op.get_folded_input_shape(i)
            shape_ref = op_ref.get_folded_input_shape(i)

            # FINN Bug: Same issue as test_normal_shapes_parity
            if i > 0 and self.get_num_inputs() > 1:
                pytest.skip(
                    f"FINN limitation: get_folded_input_shape({i}) uses broken "
                    f"get_normal_input_shape() which ignores 'ind' parameter."
                )

            assert_shapes_match(shape, shape_ref, i, "folded input")

        # Output shapes
        for i in range(self.get_num_outputs()):
            shape = op.get_folded_output_shape(i)
            shape_ref = op_ref.get_folded_output_shape(i)
            assert_shapes_match(shape, shape_ref, i, "folded output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_stream_widths_parity(self, kernel_test_config, stage2_model, stage2_model_reference):
        """Test input/output stream widths match between implementations."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        # Input stream widths
        for i in range(self.get_num_inputs()):
            width = op.get_instream_width(i)
            width_ref = op_ref.get_instream_width(i)
            assert_widths_match(width, width_ref, i, "Input")

        # Output stream widths
        for i in range(self.get_num_outputs()):
            width = op.get_outstream_width(i)
            width_ref = op_ref.get_outstream_width(i)
            assert_widths_match(width, width_ref, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_stream_widths_padded_parity(
        self, kernel_test_config, stage2_model, stage2_model_reference
    ):
        """Test padded stream widths match (AXI alignment)."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        # Input stream widths padded
        for i in range(self.get_num_inputs()):
            width = op.get_instream_width_padded(i)
            width_ref = op_ref.get_instream_width_padded(i)
            assert_values_match(
                width, width_ref, f"Input {i} stream width", lambda w: f"{w} bits (padded)"
            )

        # Output stream widths padded
        for i in range(self.get_num_outputs()):
            width = op.get_outstream_width_padded(i)
            width_ref = op_ref.get_outstream_width_padded(i)
            assert_values_match(
                width, width_ref, f"Output {i} stream width", lambda w: f"{w} bits (padded)"
            )

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_datatypes_parity(self, kernel_test_config, stage2_model, stage2_model_reference):
        """Test input/output datatypes match between implementations."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        # Input datatypes
        for i in range(self.get_num_inputs()):
            dt = op.get_input_datatype(i)
            dt_ref = op_ref.get_input_datatype(i)
            assert_datatypes_match(dt, dt_ref, i, "Input")

        # Output datatypes
        for i in range(self.get_num_outputs()):
            dt = op.get_output_datatype(i)
            dt_ref = op_ref.get_output_datatype(i)
            assert_datatypes_match(dt, dt_ref, i, "Output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_datatype_inference_parity(
        self, kernel_test_config, stage2_model, stage2_model_reference
    ):
        """Test datatype inference produces matching results.

        Note: This test compares datatypes only, not tensor names.
        Some kernels may rename tensors during infer_node_datatype(),
        so we compare datatypes by position, not name.
        """
        op, model = stage2_model
        op_ref, model_ref = stage2_model_reference

        # Run datatype inference
        model_out = op.infer_node_datatype(model)
        model_ref_out = op_ref.infer_node_datatype(model_ref)

        # Use returned model if provided
        if model_out is not None:
            model = model_out
        if model_ref_out is not None:
            model_ref = model_ref_out

        # Verify input datatypes (compare by position, not name)
        for i in range(self.get_num_inputs()):
            input_name = op.onnx_node.input[i]
            input_name_ref = op_ref.onnx_node.input[i]

            if not input_name or not input_name_ref:
                continue

            # Get datatypes from each model using its own tensor names
            dt = model.get_tensor_datatype(input_name)
            dt_ref = model_ref.get_tensor_datatype(input_name_ref)

            # Compare datatypes (names may differ)
            assert_datatypes_match(dt, dt_ref, i, "After infer_node_datatype, input")

        # Verify output datatypes (compare by position, not name)
        for i in range(self.get_num_outputs()):
            output_name = op.onnx_node.output[i]
            output_name_ref = op_ref.onnx_node.output[i]

            # Get datatypes from each model using its own tensor names
            dt = model.get_tensor_datatype(output_name)
            dt_ref = model_ref.get_tensor_datatype(output_name_ref)

            # Compare datatypes (names may differ)
            assert_datatypes_match(dt, dt_ref, i, "After infer_node_datatype, output")

    @pytest.mark.parity
    @pytest.mark.core
    @pytest.mark.kernel_parity
    def test_make_shape_compatible_op_parity(
        self, kernel_test_config, stage2_model, stage2_model_reference
    ):
        """Test shape-compatible ops preserve output structure.

        Note: make_shape_compatible_op() returns an ONNX NodeProto (per FINN API),
        not a wrapped HWCustomOp. This is used for shape inference.
        """
        op, model = stage2_model
        op_ref, model_ref = stage2_model_reference

        # Returns ONNX NodeProto for shape inference
        compat_node = op.make_shape_compatible_op(model)
        compat_node_ref = op_ref.make_shape_compatible_op(model_ref)

        # Verify output count matches (NodeProto.output is a list of output names)
        assert len(compat_node.output) == len(compat_node_ref.output), (
            f"Shape-compatible op output count mismatch: "
            f"primary={len(compat_node.output)}, "
            f"reference={len(compat_node_ref.output)}"
        )

        # Verify output names match (both should use same output names as original op)
        for i in range(self.get_num_outputs()):
            output_name = compat_node.output[i] if i < len(compat_node.output) else None
            output_name_ref = compat_node_ref.output[i] if i < len(compat_node_ref.output) else None

            assert (
                output_name == op.onnx_node.output[i]
            ), f"Primary shape-compatible op output {i} name mismatch"
            assert (
                output_name_ref == op_ref.onnx_node.output[i]
            ), f"Reference shape-compatible op output {i} name mismatch"

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_expected_cycles_parity(self, kernel_test_config, stage2_model, stage2_model_reference):
        """Test expected cycle counts match between implementations."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        cycles = op.get_exp_cycles()
        cycles_ref = op_ref.get_exp_cycles()

        assert_values_match(cycles, cycles_ref, "Expected cycles")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_number_output_values_parity(
        self, kernel_test_config, stage2_model, stage2_model_reference
    ):
        """Test number of output values match (for FIFO sizing)."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        count = op.get_number_output_values()
        count_ref = op_ref.get_number_output_values()

        assert_values_match(count, count_ref, "Number of output values")

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_resource_estimates_parity(
        self, kernel_test_config, stage2_model, stage2_model_reference
    ):
        """Test resource estimates match between implementations."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        # LUT estimation
        if hasattr(op, "lut_estimation") and hasattr(op_ref, "lut_estimation"):
            luts = op.lut_estimation()
            luts_ref = op_ref.lut_estimation()
            assert_values_match(luts, luts_ref, "LUT estimation", lambda count: f"{count:,} LUTs")

        # DSP estimation (requires fpgapart parameter per FINN API)
        if hasattr(op, "dsp_estimation") and hasattr(op_ref, "dsp_estimation"):
            # Use default fpgapart for estimation comparison
            from tests.support.constants import PARITY_DEFAULT_FPGA_PART_HLS

            fpgapart = PARITY_DEFAULT_FPGA_PART_HLS
            dsps = op.dsp_estimation(fpgapart)
            dsps_ref = op_ref.dsp_estimation(fpgapart)
            assert_values_match(dsps, dsps_ref, "DSP estimation", lambda count: f"{count:,} DSPs")

        # BRAM estimation
        if hasattr(op, "bram_estimation") and hasattr(op_ref, "bram_estimation"):
            brams = op.bram_estimation()
            brams_ref = op_ref.bram_estimation()
            assert_values_match(
                brams, brams_ref, "BRAM estimation", lambda count: f"{count:,} BRAMs"
            )

        # URAM estimation
        if hasattr(op, "uram_estimation") and hasattr(op_ref, "uram_estimation"):
            urams = op.uram_estimation()
            urams_ref = op_ref.uram_estimation()
            assert_values_match(
                urams, urams_ref, "URAM estimation", lambda count: f"{count:,} URAMs"
            )

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_efficiency_metrics_parity(
        self, kernel_test_config, stage2_model, stage2_model_reference
    ):
        """Test BRAM/URAM efficiency estimates match."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        # BRAM efficiency
        if hasattr(op, "bram_efficiency_estimation") and hasattr(
            op_ref, "bram_efficiency_estimation"
        ):
            eff = op.bram_efficiency_estimation()
            eff_ref = op_ref.bram_efficiency_estimation()
            assert_values_match(
                eff, eff_ref, "BRAM efficiency", lambda e: f"{e:.4f} ({e*100:.2f}%)"
            )

        # URAM efficiency
        if hasattr(op, "uram_efficiency_estimation") and hasattr(
            op_ref, "uram_efficiency_estimation"
        ):
            eff = op.uram_efficiency_estimation()
            eff_ref = op_ref.uram_efficiency_estimation()
            assert_values_match(
                eff, eff_ref, "URAM efficiency", lambda e: f"{e:.4f} ({e*100:.2f}%)"
            )

    @pytest.mark.parity
    @pytest.mark.hw_estimation
    @pytest.mark.kernel_parity
    def test_operation_counts_parity(
        self, kernel_test_config, stage2_model, stage2_model_reference
    ):
        """Test operation and parameter counts match."""
        op, _ = stage2_model
        op_ref, _ = stage2_model_reference

        if hasattr(op, "get_op_and_param_counts") and hasattr(op_ref, "get_op_and_param_counts"):
            counts = op.get_op_and_param_counts()
            counts_ref = op_ref.get_op_and_param_counts()
            assert_values_match(counts, counts_ref, "Operation and parameter counts")
