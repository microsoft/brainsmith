# lenet_custom_steps.py

import os
import logging

from brainsmith.registry import step

from qonnx.transformation.general import (
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    SortGraph,
)

from qonnx.core.datatype import DataType
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

from finn.transformation.streamline import Streamline
from finn.transformation.streamline.absorb import (
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoMultiThreshold,
)
from finn.transformation.streamline.reorder import (
    MoveScalarLinearPastInvariants,
    MakeMaxPoolNHWC,
)

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw

logger = logging.getLogger(__name__)


def _tidy(model):
    """Small safe cleanup used after major graph rewrites."""
    for trn in [
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        InferShapes(),
        InferDataTypes(),
        RemoveStaticGraphInputs(),
        RemoveUnusedTensors(),
        SortGraph(),
    ]:
        model = model.transform(trn)
    return model

def count_ops(model, op_type):
    return len(model.get_nodes_by_op_type(op_type))


def print_transposes(model, tag):
    transposes = model.get_nodes_by_op_type("Transpose")
    print(f"\n[{tag}] Transpose count = {len(transposes)}")

    for node in transposes:
        perm = None
        for attr in node.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)

        print(f"  name : {node.name}")
        print(f"  perm : {perm}")
        print(f"  input: {list(node.input)}")
        print(f"  out  : {list(node.output)}")

@step(name="lenet_pre_dataflow_cleanup")
def lenet_pre_dataflow_cleanup_step(model, cfg):
    """
    Safe cleanup before custom hardware conversion.

    This assumes Brainsmith has already run:
      cleanup
      qonnx_to_finn
      normalize_dataflow_layouts

    Do not call ConvertQONNXtoFINN here if the default qonnx_to_finn
    step is already present in the blueprint.
    """
    for trn in [
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        InferShapes(),
        FoldConstants(),
        DoubleToSingleFloat(),
        GemmToMatMul(),
        FoldTransposeIntoQuantInit(),
        InferDataTypes(),
        InferDataLayouts(),
        AbsorbConsecutiveTransposes(),
        RemoveUnusedTensors(),
        SortGraph(),
    ]:
        model = model.transform(trn)

    model = _tidy(model)
    model.save(os.path.join(cfg.output_dir, "lenet_01_pre_dataflow_cleanup.onnx"))
    return model


@step(name="lenet_streamline")
def lenet_streamline_step(model, cfg):
    """
    Streamline the FINN/QONNX graph before hardware-layer inference.
    """
    for trn in [
        MoveScalarLinearPastInvariants(),
        Streamline(),
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoMultiThreshold(),
        RemoveUnusedTensors(),
        SortGraph(),
    ]:
        model = model.transform(trn)

    model = _tidy(model)
    model.save(os.path.join(cfg.output_dir, "lenet_02_streamlined.onnx"))
    return model

@step(name="lenet_clean_transposes_before_hw")
def lenet_clean_transposes_before_hw_step(model, cfg):
    """
    Remove / absorb Transpose nodes before FINN hardware-layer inference.

    This step should run after:
        qonnx_to_finn
        normalize_dataflow_layouts

    and before:
        create_dataflow_partition
        specialize_layers
    """

    print_transposes(model, "initial")

    # ------------------------------------------------------------
    # 1. Fold Transpose into constants / quant initializers
    # ------------------------------------------------------------
    #
    # Pattern:
    #     Quant / constant initializer -> Transpose
    #
    # Becomes:
    #     reordered initializer
    #
    # This removes Transpose nodes that only exist because exported
    # weights/constants have the wrong physical order.
    model = model.transform(FoldTransposeIntoQuantInit())
    model = _tidy(model)
    print_transposes(model, "after FoldTransposeIntoQuantInit")

    # ------------------------------------------------------------
    # 2. Infer layout metadata and collapse Transpose -> Transpose
    # ------------------------------------------------------------
    #
    # Pattern:
    #     Transpose(perm=A) -> Transpose(perm=inverse(A))
    #
    # Becomes:
    #     identity, so both Transpose nodes disappear.
    model = model.transform(InferDataLayouts())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = _tidy(model)
    print_transposes(model, "after AbsorbConsecutiveTransposes pass 1")

    # ------------------------------------------------------------
    # 3. Move scalar operations past layout-invariant operations
    # ------------------------------------------------------------
    #
    # This helps expose more absorbable patterns.
    #
    # Example:
    #     Transpose -> Mul/Add -> MultiThreshold
    #
    # may become:
    #     Mul/Add -> Transpose -> MultiThreshold
    #
    # or may allow Streamline to absorb scalar ops.
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    model = _tidy(model)
    print_transposes(model, "after Streamline")

    # ------------------------------------------------------------
    # 4. Absorb Transpose into MultiThreshold where legal
    # ------------------------------------------------------------
    #
    # Pattern:
    #     Transpose -> MultiThreshold
    #
    # Becomes:
    #     MultiThreshold with thresholds/layout metadata adjusted
    #
    # This is very important after ConvertQONNXtoFINN(), because
    # quantized activations are often represented as MultiThreshold.
    model = model.transform(AbsorbTransposeIntoMultiThreshold())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = _tidy(model)
    print_transposes(model, "after AbsorbTransposeIntoMultiThreshold")

    # ------------------------------------------------------------
    # 5. Handle pooling layout only if pooling exists
    # ------------------------------------------------------------
    #
    # For your current LeNet variant, pooling is commented out and
    # stride-2 Conv is used instead, so this usually does nothing.
    #
    # Use MakeMaxPoolNHWC only if MaxPool is actually present.
    if count_ops(model, "MaxPool") > 0:
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(AbsorbConsecutiveTransposes())
        model = _tidy(model)
        print_transposes(model, "after MakeMaxPoolNHWC")

    # ------------------------------------------------------------
    # 6. Lower Conv to MatMul before FINN hardware inference
    # ------------------------------------------------------------
    #
    # Pattern:
    #     Conv
    #
    # Becomes:
    #     Im2Col / lowered-conv MatMul pattern
    #
    # This is needed before MVAU / ConvInpGen inference.
    model = model.transform(LowerConvsToMatMul())
    model = _tidy(model)
    print_transposes(model, "after LowerConvsToMatMul")

    # ------------------------------------------------------------
    # 7. Remove CNV-to-FC flatten boundary if possible
    # ------------------------------------------------------------
    #
    # Pattern:
    #     Conv output -> Flatten -> FC
    #
    # FINN often wants this boundary simplified before MVAU inference.
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = _tidy(model)
    print_transposes(model, "after RemoveCNVtoFCFlatten")

    # ------------------------------------------------------------
    # 8. Final transpose cleanup loop
    # ------------------------------------------------------------
    #
    # Some transforms expose new Transpose -> Transpose pairs, so run
    # this a few times until stable.
    prev_count = None

    for i in range(4):
        cur_count = count_ops(model, "Transpose")
        if cur_count == prev_count:
            break

        prev_count = cur_count

        model = model.transform(InferDataLayouts())
        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(AbsorbTransposeIntoMultiThreshold())
        model = _tidy(model)

        print_transposes(model, f"final cleanup loop {i}")

    # Save the debug graph before hardware inference.
    model.save(cfg.output_dir + "/lenet_after_transpose_cleanup.onnx")

    return model

@step(name="lenet_infer_hw_layers")
def lenet_infer_hw_layers_step(model, cfg):

    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    """
    Convert Conv/MatMul/MultiThreshold patterns into FINN fpgadataflow layers.

    Expected result:
      Conv/Im2Col/MatMul patterns
        → ConvolutionInputGenerator + MVAU

      Quantized activations
        → Thresholding

      Linear layers
        → MVAU
    """
    for trn in [
        LowerConvsToMatMul(),

        # Important for Conv → FC boundary in CNV/LeNet-like networks.
        RemoveCNVtoFCFlatten(),

        InferShapes(),
        InferDataTypes(),

        to_hw.InferQuantizedMatrixVectorActivation(),
        to_hw.InferThresholdingLayer(),
        to_hw.InferConvInpGen(),
        to_hw.InferDuplicateStreamsLayer(),
        to_hw.InferVectorVectorActivation(),
        to_hw.InferQuantizedMatrixVectorActivation(),
        to_hw.InferChannelwiseLinearLayer(),
        to_hw.InferLabelSelectLayer(),

        InferShapes(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        RemoveUnusedTensors(),
        SortGraph(),
    ]:
        model = model.transform(trn)

    model = _tidy(model)
    model.save(os.path.join(cfg.output_dir, "lenet_03_hw_layers.onnx"))
    return model


@step(name="lenet_specialize_remaining_hw_layers")
def lenet_specialize_remaining_hw_layers_step(model, cfg):
    """
    Run after build_hw_graph and before target_fps_parallelization.

    Purpose:
      Convert any remaining generic FINN fpgadataflow nodes, such as
      FMPadding and ConvolutionInputGenerator, into HLS/RTL backend variants.

    This avoids FINN dataflow_performance KeyError where an HLS/RTL node has
    a generic fpgadataflow predecessor that was skipped by latency analysis.
    """

    # Resolve FPGA part. Pynq-Z1 uses xc7z020clg400-1.
    if hasattr(cfg, "_resolve_fpga_part"):
        fpgapart = cfg._resolve_fpga_part()
    else:
        fpgapart = getattr(cfg, "fpga_part", "xc7z020clg400-1")

    print("\n[lenet_specialize_remaining_hw_layers] Before:")
    for node in model.graph.node:
        if node.domain == "finn.custom_op.fpgadataflow":
            print(f"  generic node still present: {node.name} :: {node.op_type}")

    model = model.transform(SpecializeLayers(fpgapart))
    model = _tidy_for_partition(model)

    print("\n[lenet_specialize_remaining_hw_layers] After:")
    remaining = []
    for node in model.graph.node:
        if node.domain == "finn.custom_op.fpgadataflow":
            remaining.append((node.name, node.op_type))
            print(f"  still generic: {node.name} :: {node.op_type}")

    if len(remaining) > 0:
        raise RuntimeError(
            "Generic fpgadataflow nodes still remain after specialization: "
            + str(remaining)
        )

    # Optional but useful: make sure cycle estimates can be attached before
    # target_fps_parallelization calls dataflow_performance.
    model = model.transform(AnnotateCycles())

    model.save(cfg.output_dir + "/13b_lenet_specialized_remaining_hw_layers.onnx")
    return model
########################################################################################################
def _is_inverse_perm(p0, p1):
    if p0 is None or p1 is None:
        return False
    if len(p0) != len(p1):
        return False
    return [p0[i] for i in p1] == list(range(len(p0)))


def _get_transpose_perm(node):
    for attr in node.attribute:
        if attr.name == "perm":
            return list(attr.ints)
    return None


def _remove_inverse_transpose_pairs(model):
    """
    Remove patterns like:
        X -> Transpose([0,3,1,2]) -> Transpose([0,2,3,1]) -> Y

    These are identity layout round-trips and must not remain between
    FINN fpgadataflow nodes before create_dataflow_partition.
    """
    graph = model.graph
    producer = {}
    consumers = {}

    for node in graph.node:
        for out_name in node.output:
            producer[out_name] = node
        for inp_name in node.input:
            consumers.setdefault(inp_name, []).append(node)

    nodes_to_remove = []

    for node in list(graph.node):
        if node.op_type != "Transpose":
            continue

        out_name = node.output[0]
        users = consumers.get(out_name, [])

        if len(users) != 1:
            continue

        next_node = users[0]

        if next_node.op_type != "Transpose":
            continue

        p0 = _get_transpose_perm(node)
        p1 = _get_transpose_perm(next_node)

        if not _is_inverse_perm(p0, p1):
            continue

        src_tensor = node.input[0]
        dst_tensor = next_node.output[0]

        # Rewire all users of the second transpose output to the source tensor.
        for user in consumers.get(dst_tensor, []):
            for i, inp in enumerate(user.input):
                if inp == dst_tensor:
                    user.input[i] = src_tensor

        nodes_to_remove.extend([node, next_node])

    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)

    return model


def _remove_1x1_transpose_before_flatten(model):
    """
    Remove the LeNet Conv-to-FC boundary pattern:

        MVAU_hls_2 output [N, 1, 1, C]
          -> Transpose [N, C, 1, 1]
          -> Reshape [N, C]
          -> MVAU_hls_3

    Since H = W = 1, the transpose does not change the flattened vector order.
    """
    graph = model.graph

    producer = {}
    consumers = {}

    for node in graph.node:
        for out_name in node.output:
            producer[out_name] = node
        for inp_name in node.input:
            consumers.setdefault(inp_name, []).append(node)

    nodes_to_remove = []

    for node in list(graph.node):
        if node.op_type != "Transpose":
            continue

        perm = _get_transpose_perm(node)

        if perm != [0, 3, 1, 2]:
            continue

        out_name = node.output[0]
        users = consumers.get(out_name, [])

        if len(users) != 1:
            continue

        reshape = users[0]

        if reshape.op_type != "Reshape":
            continue

        src_tensor = node.input[0]
        reshape_out = reshape.output[0]

        # Rewire users of Reshape output directly to the original tensor.
        for user in consumers.get(reshape_out, []):
            for i, inp in enumerate(user.input):
                if inp == reshape_out:
                    user.input[i] = src_tensor

        nodes_to_remove.extend([node, reshape])

    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)

    return model


def _tidy_for_partition(model):
    for trn in [
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        InferShapes(),
        InferDataTypes(),
        RemoveUnusedTensors(),
        SortGraph(),
    ]:
        model = model.transform(trn)
    return model

########################################################################################################


@step(name="lenet_pre_partition_cleanup")
def lenet_pre_partition_cleanup_step(model, cfg):
    """
    Must run after specialize_layers and before create_dataflow_partition.

    Purpose:
      remove ordinary Transpose/Reshape nodes trapped between FINN HW nodes.
    """

    print("\n[lenet_pre_partition_cleanup] Before cleanup:")
    print("  Transpose count:", len(model.get_nodes_by_op_type("Transpose")))
    print("  Reshape count  :", len(model.get_nodes_by_op_type("Reshape")))

    # First try standard FINN/QONNX cleanup.
    for trn in [
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoMultiThreshold(),
        RemoveCNVtoFCFlatten(),
    ]:
        model = model.transform(trn)
        model = _tidy_for_partition(model)

    # Then remove the exact LeNet blocking patterns still present after specialize_layers.
    model = _remove_inverse_transpose_pairs(model)
    model = _tidy_for_partition(model)

    model = _remove_1x1_transpose_before_flatten(model)
    model = _tidy_for_partition(model)

    # One more standard cleanup pass.
    for trn in [
        AbsorbConsecutiveTransposes(),
        RemoveCNVtoFCFlatten(),
    ]:
        model = model.transform(trn)
        model = _tidy_for_partition(model)

    print("\n[lenet_pre_partition_cleanup] After cleanup:")
    print("  Transpose count:", len(model.get_nodes_by_op_type("Transpose")))
    print("  Reshape count  :", len(model.get_nodes_by_op_type("Reshape")))

    model.save(cfg.output_dir + "/10b_lenet_pre_partition_cleanup.onnx")
    return model

@step(name="lenet_partition_sanity_check")
def lenet_partition_sanity_check_step(model, cfg):
    """
    Save the graph immediately before Brainsmith/FINN partitioning.
    This is useful for Netron inspection.
    """
    model = _tidy(model)
    model.save(os.path.join(cfg.output_dir, "lenet_04_before_partition.onnx"))
    return model
