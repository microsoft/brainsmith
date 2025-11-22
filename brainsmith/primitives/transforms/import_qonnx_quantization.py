# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Import quantization metadata from QONNX format.

This transform prepares QONNX models for Brainsmith hardware compilation by:
1. Folding quantization into weight initializers
2. Converting activation quantization nodes (Quant, BipolarQuant) to MultiThreshold
3. Converting AvgPool+Trunc patterns to QuantAvgPool2d

This transform handles ONLY quantization-specific operations. Topology transformations
(GemmToMatMul, ExtractBiasFromConv, etc.) belong in finn_topology_cleanup_step.

Similar transforms can be added for other quantization frameworks
(e.g., ImportTensorRTQuantization, ImportPyTorchQuantization).
"""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes

from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.infer_quant_avg_pool_2d import AvgPoolAndTruncToQuantAvgPool
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    ConvertQuantActToMultiThreshold,
    default_filter_function_generator,
)


class ImportQONNXQuantization(Transformation):
    """Import QONNX quantization metadata for Brainsmith.

    Handles ONLY quantization-specific transforms:
    1. FoldQuantWeights - Fold quantization into weight initializers
    2. ConvertQuantActToMultiThreshold - Convert Quant/BipolarQuant to MultiThreshold
    3. AvgPoolAndTruncToQuantAvgPool - Convert AvgPool+Trunc pattern to QuantAvgPool2d

    Topology transforms (GemmToMatMul, ExtractBiasFromConv, etc.) belong in
    finn_topology_cleanup_step, not here.

    Should be run after topology cleanup, before streamlining.
    """

    def __init__(
        self,
        filter_function=default_filter_function_generator(max_multithreshold_bit_width=8),
    ):
        super().__init__()
        self._filter_function = filter_function

    def apply(self, model: ModelWrapper):
        """Apply QONNX quantization import.

        Args:
            model: QONNX ModelWrapper (after topology cleanup)

        Returns:
            Tuple of (transformed_model, graph_modified)
        """
        model = model.transform(InferDataTypes())
        model = model.transform(FoldQuantWeights())
        model = model.transform(
            ConvertQuantActToMultiThreshold(filter_function=self._filter_function)
        )
        model = model.transform(InferDataTypes())
        model = model.transform(AvgPoolAndTruncToQuantAvgPool())

        return model, False
