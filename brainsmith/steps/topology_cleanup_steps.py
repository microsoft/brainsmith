# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Topology Cleanup Steps

Initial graph topology transformations that prepare models for quantization import.
These steps normalize graph structure before quantization metadata is processed.
"""

import logging
from typing import Any

from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.general import ConvertDivToMul
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps

from brainsmith.primitives.transforms.import_qonnx_quantization import ImportQONNXQuantization
from brainsmith.registry import step

logger = logging.getLogger(__name__)


@step(name="finn_topology_cleanup")
def finn_topology_cleanup_step(model: Any, cfg: Any) -> Any:
    """Generic graph topology cleanup for FINN compatibility.

    Applies structural transformations to normalize the graph:
    - ExtractBiasFromConv: Decompose Conv with bias into Conv + Add
    - GemmToMatMul: Convert Gemm to MatMul (FINN doesn't support Gemm)
    - FoldTransposeIntoQuantInit: Fold Transpose into weight initializers
    - FoldConstants: Constant propagation and folding
    - ConvertDivToMul: Normalize division to multiplication
    - RemoveIdentityOps: Remove no-op nodes
    """
    for transform in [
        ExtractBiasFromConv(),
        GemmToMatMul(),
        FoldTransposeIntoQuantInit(),
        FoldConstants(),
        ConvertDivToMul(),
        RemoveIdentityOps(),
    ]:
        model = model.transform(transform)
    return model


@step(name="import_qonnx_quantization")
def import_qonnx_quantization_step(model: Any, cfg: Any) -> Any:
    """Import QONNX quantization metadata for hardware compilation.

    Converts QONNX quantization nodes (Quant, BipolarQuant, Trunc) to FINN
    quantization representation (MultiThreshold, QuantAvgPool2d) and prepares
    threshold values for integer hardware.
    """
    model = model.transform(ImportQONNXQuantization())
    return model
