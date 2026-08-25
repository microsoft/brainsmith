import os
import sys
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.export import export_qonnx

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model

from qonnx.transformation.general import (
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    SortGraph,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit

# ---------------------------------------------------------------------
# IMPORTANT:
# Add this script directory to PYTHONPATH and import custom steps
# before Brainsmith parses the blueprint.
# ---------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import lenet_custom_steps  # noqa: F401

from brainsmith.registry import has_step
from brainsmith.dse.api import explore_design_space

class QuantLeNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        weight_bit_width=8,
        act_bit_width=8,
    ):
        super().__init__()

        self.quant_in = qnn.QuantIdentity(
            bit_width=act_bit_width,
            return_quant_tensor=True,
        )

        self.conv1 = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=6,
            kernel_size=5,
            padding=2,
            stride=2,
            bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True,
        )

        self.act1 = qnn.QuantReLU(
            bit_width=act_bit_width,
            return_quant_tensor=True,
        )

        self.conv2 = qnn.QuantConv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=2,
            bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True,
        )

        self.act2 = qnn.QuantReLU(
            bit_width=act_bit_width,
            return_quant_tensor=True,
        )

        self.conv3 = qnn.QuantConv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True,
        )

        self.act3 = qnn.QuantReLU(
            bit_width=act_bit_width,
            return_quant_tensor=True,
        )

        self.fc1 = qnn.QuantLinear(
            in_features=120,
            out_features=84,
            bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True,
        )

        self.act4 = qnn.QuantReLU(
            bit_width=act_bit_width,
            return_quant_tensor=True,
        )

        self.fc2 = qnn.QuantLinear(
            in_features=84,
            out_features=num_classes,
            bias=False,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=False,
        )

    def forward(self, x):
        x = self.quant_in(x)

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.act3(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)

        return x


def assert_custom_steps_registered():
    required_steps = [
        "lenet_pre_dataflow_cleanup",
        "lenet_streamline",
        "lenet_clean_transposes_before_hw",
        "lenet_infer_hw_layers",
        "lenet_partition_sanity_check",
        "lenet_pre_partition_cleanup",
        "lenet_specialize_remaining_hw_layers",
    ]

    print("\nChecking custom Brainsmith step registration:")

    for step_name in required_steps:
        ok = has_step(step_name)
        print(f"  {step_name}: {ok}")

        if not ok:
            raise RuntimeError(
                f"Custom step '{step_name}' was not registered. "
                "Make sure lenet_custom_steps.py is importable and imported "
                "before Brainsmith parses the blueprint."
            )


def generate_lenet_qonnx(output_dir):
    raw_qonnx_path = os.path.join(output_dir, "lenet_raw.onnx")
    clean_qonnx_path = os.path.join(output_dir, "lenet_clean.onnx")

    model = QuantLeNet(
        in_channels=1,
        num_classes=10,
        weight_bit_width=8,
        act_bit_width=8,
    )
    model.eval()

    dummy_input = torch.rand(1, 1, 28, 28)

    print("\nExporting raw LeNet QONNX:")
    print("  Raw QONNX:", raw_qonnx_path)

    export_qonnx(
        model,
        input_t=dummy_input,
        export_path=raw_qonnx_path,
    )

    print("\nCleaning QONNX model:")
    print("  Clean QONNX:", clean_qonnx_path)

    onnx_model = ModelWrapper(raw_qonnx_path)
    onnx_model = cleanup_model(onnx_model)

    for trn in [
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        GiveUniqueParameterTensors(),

        InferShapes(),
        FoldConstants(),
        DoubleToSingleFloat(),
        GemmToMatMul(),
        FoldTransposeIntoQuantInit(),

        InferDataTypes(),

        RemoveStaticGraphInputs(),
        RemoveUnusedTensors(),
        SortGraph(),

        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        InferShapes(),
        InferDataTypes(),
    ]:
        onnx_model = onnx_model.transform(trn)

    onnx_model.save(clean_qonnx_path)

    print("Saved clean LeNet QONNX:", clean_qonnx_path)

    return clean_qonnx_path


def run_brainsmith_dse(model_path, args):
    blueprint_path = Path(args.blueprint).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("\nRunning Brainsmith DSE / FINN build:")
    print("  Model    :", model_path)
    print("  Blueprint:", blueprint_path)
    print("  Output   :", output_dir)

    explore_design_space(
        model_path=str(Path(model_path).resolve()),
        blueprint_path=str(blueprint_path),
        output_dir=str(output_dir),
    )


def main():
    parser = argparse.ArgumentParser("LeNet-5 Brainsmith demo")

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory name or path",
    )

    parser.add_argument(
        "--blueprint",
        default="lenet.yaml",
        help="Blueprint YAML file",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory before building",
    )

    args = parser.parse_args()

    # Resolve output directory.
    # This keeps behavior simple and predictable.
    args.output_dir = os.path.abspath(args.output)

    if args.force and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Custom steps must be registered before Brainsmith parses YAML.
    assert_custom_steps_registered()

    print("\nOutput:", args.output_dir)
    print("Blueprint:", args.blueprint)

    print("\nStep 1: Generate cleaned LeNet QONNX input")
    model_path = generate_lenet_qonnx(args.output_dir)

    print("\nStep 2: Run Brainsmith DSE / FINN build")
    run_brainsmith_dse(model_path, args)

    print("\nBUILD COMPLETED")


if __name__ == "__main__":
    main()
