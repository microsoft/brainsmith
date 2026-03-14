"""Parity tests for Thresholding kernel (Brainsmith vs FINN).

Compares Brainsmith's schema-driven Thresholding implementation against
FINN's traditional Thresholding kernel across both HLS and RTL backends.

Test coverage (18 inherited tests × N configurations):
- Core parity: shapes, datatypes, stream widths
- HW estimation: cycles, resources, efficiency
- Golden execution: python/cppsim/rtlsim for both implementations

Note: Brainsmith removed runtime_writeable_weights support, so we skip
those test cases.
"""

import pytest
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.kernels.thresholding.thresholding import Thresholding
from tests.fixtures.model_builders import make_multithreshold_model
from tests.frameworks.kernel_parity_test import KernelParityTest
from tests.frameworks.test_config import (
    DesignParameters,
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
)


class TestThresholdingParity(KernelParityTest):
    """Test parity between Brainsmith Thresholding and FINN Thresholding.

    Validates that Brainsmith's schema-driven implementation produces
    identical results to FINN's traditional implementation across:
    - Multiple quantization configurations (INT8→UINT4, BIPOLAR)
    - Different parallelization factors (PE=4, PE=16)
    - Both HLS and RTL backends
    """

    # ========================================================================
    # Test Configurations
    # ========================================================================

    @pytest.fixture(
        params=[
            # =================================================================
            # CATEGORY 1: Output Datatype Edge Cases
            # =================================================================
            # UINT2: Minimum threshold count (3 thresholds)
            KernelTestConfig(
                test_id="dtype_uint2_min_thresholds",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT2"]},
                    dimensions={"thresh_shape": (64, 3), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # UINT4: Standard 4-bit unsigned (15 thresholds)
            KernelTestConfig(
                test_id="dtype_uint4_standard",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 28, 28, 128)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (128, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 16}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # UINT8: Maximum threshold count (255 thresholds)
            KernelTestConfig(
                test_id="dtype_uint8_max_thresholds",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 64)},
                    input_dtypes={"inp": DataType["INT16"]},
                    output_dtypes={"out": DataType["UINT8"]},
                    dimensions={"thresh_shape": (64, 255), "thresh_dtype": "INT16"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # INT4: Signed output (requires ActVal=-8)
            KernelTestConfig(
                test_id="dtype_int4_signed_output",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["INT4"]},
                    dimensions={"thresh_shape": (64, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # BIPOLAR: Binary classification (-1/+1 output)
            KernelTestConfig(
                test_id="dtype_bipolar_binary",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["BIPOLAR"]},
                    dimensions={"thresh_shape": (64, 1), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # =================================================================
            # CATEGORY 2: PE Configuration Edge Cases
            # =================================================================
            # PE = 1: Maximum folding (sequential processing)
            KernelTestConfig(
                test_id="pe_1_max_folding",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (64, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 1}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # PE = channels: Full parallelism (unrolls to FFs)
            KernelTestConfig(
                test_id="pe_equals_channels_full_parallel",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 32)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (32, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 32}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # PE = 4: Low parallelism
            KernelTestConfig(
                test_id="pe_4_low_parallel",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 14, 14, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (64, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 4}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # PE = 32: High parallelism with large channels
            KernelTestConfig(
                test_id="pe_32_high_parallel",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 14, 14, 256)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (256, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 32}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # =================================================================
            # CATEGORY 3: Per-Tensor Quantization (Threshold Broadcasting)
            # =================================================================
            # Per-tensor UINT4 with PE=8
            KernelTestConfig(
                test_id="pertensor_uint4_pe8",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (1, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # Per-tensor with PE=1 (maximum folding + broadcasting)
            KernelTestConfig(
                test_id="pertensor_pe1_max_folding",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 32)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (1, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 1}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # Per-tensor with PE=channels (full parallel + broadcasting)
            KernelTestConfig(
                test_id="pertensor_pe_equals_channels",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 16)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (1, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 16}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # Per-tensor BIPOLAR
            KernelTestConfig(
                test_id="pertensor_bipolar",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 32)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["BIPOLAR"]},
                    dimensions={"thresh_shape": (1, 1), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # Per-tensor with large channel count (BERT-like)
            # NOTE: 3D inputs not supported by QONNX MultiThreshold.execute_node()
            # Uncomment when QONNX adds 3D support
            # KernelTestConfig(
            #     test_id="pertensor_large_channels_bert",
            #     model=ModelStructure(
            #         operation="MultiThreshold",
            #         input_shapes={"inp": (1, 32, 128)},
            #         input_dtypes={"inp": DataType["INT8"]},
            #         output_dtypes={"out": DataType["UINT4"]},
            #         dimensions={"thresh_shape": (1, 15), "thresh_dtype": "INT8"},
            #     ),
            #     design=DesignParameters(input_streams={0: 16}),
            #     platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            # ),
            # Per-tensor UINT2 (minimum thresholds + broadcasting)
            KernelTestConfig(
                test_id="pertensor_uint2_min",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT2"]},
                    dimensions={"thresh_shape": (1, 3), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # =================================================================
            # CATEGORY 4: Input Dimension Variations
            # =================================================================
            # 2D input: FC layer output (batch, features)
            KernelTestConfig(
                test_id="dim_2d_fc_like",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 128)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (128, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 16}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # 3D input: Sequence model (batch, seq, features)
            # NOTE: 3D inputs not supported by QONNX MultiThreshold.execute_node()
            # Uncomment when QONNX adds 3D support
            # KernelTestConfig(
            #     test_id="dim_3d_sequence",
            #     model=ModelStructure(
            #         operation="MultiThreshold",
            #         input_shapes={"inp": (1, 64, 128)},
            #         input_dtypes={"inp": DataType["INT8"]},
            #         output_dtypes={"out": DataType["UINT4"]},
            #         dimensions={"thresh_shape": (128, 15), "thresh_dtype": "INT8"},
            #     ),
            #     design=DesignParameters(input_streams={0: 16}),
            #     platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            # ),
            # 4D non-square spatial dimensions
            KernelTestConfig(
                test_id="dim_4d_nonsquare",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 28, 14, 128)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (128, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 16}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # 4D small spatial (edge case)
            KernelTestConfig(
                test_id="dim_4d_small_spatial",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 1, 1, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (64, 15), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # =================================================================
            # CATEGORY 5: Input Datatype Variations
            # =================================================================
            # UINT8 input (unsigned, non-negative thresholds)
            KernelTestConfig(
                test_id="input_uint8_unsigned",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["UINT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (64, 15), "thresh_dtype": "UINT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # INT16 input (wider datapath)
            KernelTestConfig(
                test_id="input_int16_wide",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 64)},
                    input_dtypes={"inp": DataType["INT16"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (64, 15), "thresh_dtype": "INT16"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # =================================================================
            # CATEGORY 6: Narrow Range Quantization
            # =================================================================
            # Narrow UINT4: 14 thresholds instead of 15
            KernelTestConfig(
                test_id="narrow_uint4_14_thresholds",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (64, 14), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
            # Narrow per-tensor (broadcasting + narrow range)
            KernelTestConfig(
                test_id="narrow_pertensor_14_thresholds",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 32)},
                    input_dtypes={"inp": DataType["INT8"]},
                    output_dtypes={"out": DataType["UINT4"]},
                    dimensions={"thresh_shape": (1, 14), "thresh_dtype": "INT8"},
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xc7z020clg400-1"),
            ),
        ]
    )
    def kernel_test_config(self, request):
        """Provide test configurations for Thresholding parity tests."""
        return request.param

    # ========================================================================
    # Required Abstract Methods - Primary Implementation (Brainsmith)
    # ========================================================================

    def get_kernel_op(self):
        """Return Brainsmith Thresholding class for primary implementation."""
        return Thresholding

    # ========================================================================
    # Required Abstract Methods - Reference Implementation (FINN)
    # ========================================================================

    def infer_kernel_reference(
        self,
        model: ModelWrapper,
        target_node: str,
    ) -> tuple[HWCustomOp, ModelWrapper]:
        """Infer reference kernel using FINN InferThresholdingLayer.

        Applies FINN's transformation pipeline:
        1. InferThresholdingLayer: MultiThreshold → Thresholding
        2. Dtype optimization transforms (match Brainsmith's VALUE_OPTIMIZED)
        3. Find and return Thresholding node

        Args:
            model: Stage 1 model (ONNX with annotations)
            target_node: Target node name (unused - FINN doesn't preserve names)

        Returns:
            (op, model): FINN Thresholding kernel and transformed model
        """
        from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer
        from finn.transformation.fpgadataflow.minimize_accumulator_width import (
            MinimizeAccumulatorWidth,
        )
        from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
            MinimizeWeightBitWidth,
        )
        from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
        from qonnx.custom_op.registry import getCustomOp
        from qonnx.transformation.infer_datatypes import InferDataTypes

        # Apply FINN transformation pipeline
        model = model.transform(InferThresholdingLayer())

        # Apply dtype optimizations to match Brainsmith's VALUE_OPTIMIZED behavior
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(RoundAndClipThresholds())
        model = model.transform(InferDataTypes())

        # FINN doesn't preserve node names during transformation
        # Find Thresholding node by op_type
        nodes_by_op_type = model.get_nodes_by_op_type("Thresholding")
        assert len(nodes_by_op_type) == 1, (
            f"Expected 1 Thresholding node after InferThresholdingLayer, "
            f"found {len(nodes_by_op_type)}"
        )

        onnx_node = nodes_by_op_type[0]
        op = getCustomOp(onnx_node)

        return op, model

    def get_backend_variants_reference(self) -> list[type]:
        """Return FINN backend variants (HLS and RTL).

        Returns:
            List containing FINN's Thresholding_hls backend class
        """
        from finn.custom_op.fpgadataflow.hls.thresholding_hls import Thresholding_hls

        # Note: Could also test RTL backend:
        # from finn.custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl
        # return [Thresholding_rtl]

        return [Thresholding_hls]

    # ========================================================================
    # Required Abstract Methods - Validation Counts
    # ========================================================================

    def get_num_inputs(self) -> int:
        """Thresholding has 1 dynamic input (data), thresholds are static."""
        return 1

    def get_num_outputs(self) -> int:
        """Thresholding has 1 output."""
        return 1

    # ========================================================================
    # Model Builder
    # ========================================================================

    def make_test_model(
        self,
        kernel_test_config: KernelTestConfig,
    ) -> tuple[ModelWrapper, list[str]]:
        """Create ONNX model with MultiThreshold node.

        Uses make_multithreshold_model() helper to generate a properly
        configured MultiThreshold node with evenly-spaced threshold values.

        Args:
            kernel_test_config: Test configuration with shapes/dtypes

        Returns:
            (model, input_names): ONNX model and list of input tensor names
        """
        model_struct = kernel_test_config.model

        # Extract configuration
        inp_shape = model_struct.input_shapes["inp"]

        # Threshold config comes from dimensions (static weight, not dynamic input)
        thresh_shape = model_struct.dimensions.get("thresh_shape", (inp_shape[-1], 15))

        inp_dtype_str = model_struct.input_dtypes["inp"].name
        thresh_dtype_str = model_struct.dimensions.get("thresh_dtype", "INT8")

        # Get output dtype from model structure (check both output_dtypes and dimensions)
        if model_struct.output_dtypes and "out" in model_struct.output_dtypes:
            out_dtype = model_struct.output_dtypes["out"]
            out_dtype_str = out_dtype.name
        elif model_struct.dimensions and "output_dtype" in model_struct.dimensions:
            out_dtype_str = model_struct.dimensions["output_dtype"]
            out_dtype = DataType[out_dtype_str]
        else:
            out_dtype_str = "UINT4"
            out_dtype = DataType[out_dtype_str]

        # Compute num_thresholds from thresh_shape
        num_channels = thresh_shape[0]
        num_thresholds = thresh_shape[1]

        # Determine out_bias (ActVal) based on output datatype
        # For signed outputs (except BIPOLAR), out_bias should be negative
        if out_dtype != DataType["BIPOLAR"] and out_dtype.signed():
            # Signed output: ActVal should be negative
            # For UINT4 (15 thresholds), ActVal = 0
            # For INT4 (15 thresholds), ActVal = -8
            out_bias = -(2 ** (out_dtype.bitwidth() - 1))
        else:
            # Unsigned or BIPOLAR: ActVal = 0
            out_bias = 0

        # Create MultiThreshold model
        model, node = make_multithreshold_model(
            shape=list(inp_shape),
            input_dtype=inp_dtype_str,
            threshold_dtype=thresh_dtype_str,
            output_dtype=out_dtype_str,
            num_thresholds=num_thresholds,
            out_scale=1.0,
            out_bias=out_bias,
        )

        # Return model and input names
        # Only dynamic input "inp" - thresholds are static initializer
        input_names = [node.input[0]]  # ["inp"] only

        return model, input_names


# =============================================================================
# RTL Backend Parity Tests
# =============================================================================


class TestThresholdingParityRTL(TestThresholdingParity):
    """Test parity for Thresholding RTL backend.

    Inherits all test configurations from TestThresholdingParity but uses
    RTL backend variants instead of HLS.
    """

    def get_backend_variants(self) -> list[type]:
        """Return Brainsmith RTL backend variant."""
        from brainsmith.kernels.thresholding.thresholding_rtl import Thresholding_rtl

        return [Thresholding_rtl]

    def get_backend_variants_reference(self) -> list[type]:
        """Return FINN RTL backend variant."""
        from finn.custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl

        return [Thresholding_rtl]

    @pytest.fixture(
        params=[
            # =================================================================
            # CATEGORY 1: Output Datatype Edge Cases (RTL)
            # =================================================================
            # UINT2: Minimum threshold count
            KernelTestConfig(
                test_id="rtl_dtype_uint2_min",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (64, 3),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT2"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # UINT4: Standard
            KernelTestConfig(
                test_id="rtl_dtype_uint4_standard",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 28, 28, 128)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (128, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 16}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # UINT8: Maximum thresholds
            KernelTestConfig(
                test_id="rtl_dtype_uint8_max",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 64)},
                    input_dtypes={"inp": DataType["INT16"]},
                    dimensions={
                        "thresh_shape": (64, 255),
                        "thresh_dtype": "INT16",
                        "output_dtype": "UINT8"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # INT4: Signed output
            KernelTestConfig(
                test_id="rtl_dtype_int4_signed",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (64, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "INT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # BIPOLAR
            KernelTestConfig(
                test_id="rtl_dtype_bipolar",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (64, 1),
                        "thresh_dtype": "INT8",
                        "output_dtype": "BIPOLAR"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # =================================================================
            # CATEGORY 2: PE Configuration Edge Cases (RTL)
            # =================================================================
            # PE = 1: Maximum folding
            KernelTestConfig(
                test_id="rtl_pe_1_max_folding",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (64, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 1}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # PE = channels: Full parallelism
            KernelTestConfig(
                test_id="rtl_pe_equals_channels",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 32)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (32, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 32}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # PE = 32: High parallelism
            KernelTestConfig(
                test_id="rtl_pe_32_high_parallel",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 14, 14, 256)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (256, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 32}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # =================================================================
            # CATEGORY 3: Per-Tensor Quantization (RTL)
            # =================================================================
            # Per-tensor UINT4 with PE=8
            KernelTestConfig(
                test_id="rtl_pertensor_uint4_pe8",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (1, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # Per-tensor with PE=1 (max folding + broadcasting)
            KernelTestConfig(
                test_id="rtl_pertensor_pe1",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 32)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (1, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 1}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # Per-tensor with PE=channels (full parallel + broadcasting)
            KernelTestConfig(
                test_id="rtl_pertensor_pe_equals_channels",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 16)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (1, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 16}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # Per-tensor BIPOLAR
            KernelTestConfig(
                test_id="rtl_pertensor_bipolar",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 32)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (1, 1),
                        "thresh_dtype": "INT8",
                        "output_dtype": "BIPOLAR"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # Per-tensor with large channels (BERT-like)
            # NOTE: 3D inputs not supported by QONNX MultiThreshold.execute_node()
            # KernelTestConfig(
            #     test_id="rtl_pertensor_bert_large",
            #     model=ModelStructure(
            #         operation="MultiThreshold",
            #         input_shapes={"inp": (1, 32, 128)},
            #         input_dtypes={"inp": DataType["INT8"]},
            #         dimensions={
            #             "thresh_shape": (1, 15),
            #             "thresh_dtype": "INT8",
            #             "output_dtype": "UINT4"
            #         },
            #     ),
            #     design=DesignParameters(input_streams={0: 16}),
            #     platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            # ),
            # Per-tensor UINT2 (minimum thresholds + broadcasting)
            KernelTestConfig(
                test_id="rtl_pertensor_uint2",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (1, 3),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT2"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # =================================================================
            # CATEGORY 4: Input Dimension Variations (RTL)
            # =================================================================
            # 2D input: FC layer
            KernelTestConfig(
                test_id="rtl_dim_2d_fc",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 128)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (128, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 16}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # 3D input: Sequence model
            # NOTE: 3D inputs not supported by QONNX MultiThreshold.execute_node()
            # KernelTestConfig(
            #     test_id="rtl_dim_3d_sequence",
            #     model=ModelStructure(
            #         operation="MultiThreshold",
            #         input_shapes={"inp": (1, 64, 128)},
            #         input_dtypes={"inp": DataType["INT8"]},
            #         dimensions={
            #             "thresh_shape": (128, 15),
            #             "thresh_dtype": "INT8",
            #             "output_dtype": "UINT4"
            #         },
            #     ),
            #     design=DesignParameters(input_streams={0: 16}),
            #     platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            # ),
            # 4D non-square
            KernelTestConfig(
                test_id="rtl_dim_4d_nonsquare",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 28, 14, 128)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (128, 15),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 16}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # =================================================================
            # CATEGORY 5: Narrow Range Quantization (RTL)
            # =================================================================
            # Narrow UINT4: 14 thresholds
            KernelTestConfig(
                test_id="rtl_narrow_uint4",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 16, 16, 64)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (64, 14),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
            # Narrow per-tensor (broadcasting + narrow range)
            KernelTestConfig(
                test_id="rtl_narrow_pertensor",
                model=ModelStructure(
                    operation="MultiThreshold",
                    input_shapes={"inp": (1, 8, 8, 32)},
                    input_dtypes={"inp": DataType["INT8"]},
                    dimensions={
                        "thresh_shape": (1, 14),
                        "thresh_dtype": "INT8",
                        "output_dtype": "UINT4"
                    },
                ),
                design=DesignParameters(input_streams={0: 8}),
                platform=PlatformConfig(fpgapart="xczu3eg-sbva484-1-e"),
            ),
        ]
    )
    def kernel_test_config(self, request):
        """Provide RTL-specific test configurations."""
        return request.param


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "parity"])
