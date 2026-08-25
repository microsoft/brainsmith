#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


# ---------------------------------------------------------------------
# Folding config block generators
# ---------------------------------------------------------------------

def mvau_config(simd: int, pe: int, runtime_writeable_weights: int = 0) -> dict:
    return {
        "PE": int(pe),
        "SIMD": int(simd),
        "ram_style": "auto",
        "resType": "auto",
        "mem_mode": "internal_decoupled",
        "runtime_writeable_weights": int(runtime_writeable_weights),
    }


def thresholding_config(pe: int, runtime_writeable_weights: int = 0) -> dict:
    return {
        "PE": int(pe),
        "runtime_writeable_weights": int(runtime_writeable_weights),
        "depth_trigger_uram": 0,
        "depth_trigger_bram": 0,
    }


def convinpgen_config(simd: int) -> dict:
    # ConvolutionInputGenerator normally needs its SIMD to be compatible
    # with the following MVAU's SIMD.
    return {
        "SIMD": int(simd),
    }


def labelselect_config(pe: int) -> dict:
    return {
        "PE": int(pe),
    }


def generic_pe_config(pe: int) -> dict:
    return {
        "PE": int(pe),
        "ram_style": "auto",
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def node_text(node) -> str:
    return f"{node.name or ''} {node.op_type or ''}"


def is_mvau(node) -> bool:
    return "MVAU" in node_text(node)


def is_thresholding(node) -> bool:
    return "Thresholding" in node_text(node)


def is_convinpgen(node) -> bool:
    text = node_text(node)
    return "ConvolutionInputGenerator" in text or "ConvInpGen" in text


def is_labelselect(node) -> bool:
    return "LabelSelect" in node_text(node)


def is_channelwise_or_elementwise(node) -> bool:
    text = node_text(node)
    return any(
        key in text
        for key in [
            "ChannelwiseOp",
            "Elementwise",
            "ElementWise",
            "AddStreams",
            "DuplicateStreams",
        ]
    )


def divisors(n: int) -> list[int]:
    if n <= 0:
        return [1]
    return [d for d in range(1, n + 1) if n % d == 0]


def choose_divisor(n: Optional[int], preferred: int) -> int:
    if n is None or n <= 0:
        return 1

    valid = [d for d in divisors(n) if d <= preferred]
    return max(valid) if valid else 1


def safe_get_custom_op(node):
    try:
        return getCustomOp(node)
    except Exception:
        return None


def get_attr(node, *names: str) -> Optional[Any]:
    inst = safe_get_custom_op(node)
    if inst is None:
        return None

    try:
        attr_types = inst.get_nodeattr_types()
    except Exception:
        attr_types = {}

    for name in names:
        if name in attr_types:
            try:
                return inst.get_nodeattr(name)
            except Exception:
                pass

    return None


def get_int_attr(node, *names: str) -> Optional[int]:
    value = get_attr(node, *names)

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def get_last_non_one_dim(model: ModelWrapper, tensor_name: str) -> Optional[int]:
    try:
        shape = model.get_tensor_shape(tensor_name)
    except Exception:
        return None

    if shape is None:
        return None

    dims = []
    for dim in shape:
        try:
            dim_int = int(dim)
            if dim_int != 1:
                dims.append(dim_int)
        except Exception:
            pass

    if not dims:
        return None

    return dims[-1]


def build_graph_maps(model: ModelWrapper):
    producer = {}
    consumers = {}

    for node in model.graph.node:
        for out_name in node.output:
            producer[out_name] = node

        for inp_name in node.input:
            consumers.setdefault(inp_name, []).append(node)

    return producer, consumers


def find_direct_successors(node, consumers):
    successors = []

    for out_name in node.output:
        successors.extend(consumers.get(out_name, []))

    return successors


def infer_mvau_dims(node) -> tuple[Optional[int], Optional[int]]:
    # FINN MVAU commonly uses MW/MH, sometimes MatrixW/MatrixH.
    mw = get_int_attr(node, "MW", "MatrixW")
    mh = get_int_attr(node, "MH", "MatrixH")
    return mw, mh


def infer_channel_count(model: ModelWrapper, node) -> Optional[int]:
    # Try common FINN attributes first.
    ch = get_int_attr(
        node,
        "NumChannels",
        "numChannels",
        "Channels",
        "num_channels",
        "MH",
        "MatrixH",
    )

    if ch is not None:
        return ch

    # Fallback to output tensor shape.
    if len(node.output) > 0:
        return get_last_non_one_dim(model, node.output[0])

    return None


def estimate_mvau_cycles(mw: Optional[int], mh: Optional[int], simd: int, pe: int) -> Optional[int]:
    if mw is None or mh is None or simd <= 0 or pe <= 0:
        return None

    if mw % simd != 0 or mh % pe != 0:
        return None

    return (mw // simd) * (mh // pe)


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------

def generate_config(args) -> dict:
    model = ModelWrapper(args.model)
    producer, consumers = build_graph_maps(model)

    config = {
        "Defaults": {}
    }

    report = {
        "mvau": [],
        "thresholding": [],
        "convinpgen": [],
        "labelselect": [],
        "generic_pe": [],
        "skipped": [],
    }

    # First pass: configure all MVAUs.
    # This gives us MVAU SIMD/PE values that ConvInpGen and Thresholding can match.
    mvau_by_name = {}

    for node in model.graph.node:
        if not is_mvau(node):
            continue

        if not node.name:
            report["skipped"].append((node.name, node.op_type, "unnamed MVAU"))
            continue

        mw, mh = infer_mvau_dims(node)

        simd = choose_divisor(mw, args.simd)
        pe = choose_divisor(mh, args.pe)

        # Optional LeNet-specific safeguard:
        # avoid PE > output channels and SIMD > input vector width.
        config[node.name] = mvau_config(
            simd=simd,
            pe=pe,
            runtime_writeable_weights=args.runtime_writeable_weights,
        )

        approx_cycles = estimate_mvau_cycles(mw, mh, simd, pe)

        mvau_by_name[node.name] = {
            "node": node,
            "MW": mw,
            "MH": mh,
            "SIMD": simd,
            "PE": pe,
        }

        report["mvau"].append(
            {
                "name": node.name,
                "op_type": node.op_type,
                "MW": mw,
                "MH": mh,
                "SIMD": simd,
                "PE": pe,
                "approx_cycles": approx_cycles,
            }
        )

    # Second pass: configure the non-MVAU HW nodes around the MVAUs.
    for node in model.graph.node:
        if not node.name:
            continue

        if is_mvau(node):
            continue

        text = node_text(node)

        if is_thresholding(node):
            ch = infer_channel_count(model, node)
            pe = choose_divisor(ch, args.threshold_pe)

            # If Thresholding directly follows an MVAU, use the same PE when legal.
            pred_nodes = [producer[inp] for inp in node.input if inp in producer]
            for pred in pred_nodes:
                if pred.name in mvau_by_name:
                    mvau_pe = mvau_by_name[pred.name]["PE"]
                    if ch is None or ch % mvau_pe == 0:
                        pe = mvau_pe

            config[node.name] = thresholding_config(
                pe=pe,
                runtime_writeable_weights=0,
            )

            report["thresholding"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "channels": ch,
                    "PE": pe,
                }
            )

        elif is_convinpgen(node):
            ifm_ch = get_int_attr(node, "IFMChannels", "Channels", "NumChannels")
            simd = choose_divisor(ifm_ch, args.convinpgen_simd)

            # If ConvInpGen directly feeds an MVAU, match that MVAU's SIMD.
            for succ in find_direct_successors(node, consumers):
                if succ.name in mvau_by_name:
                    mvau_simd = mvau_by_name[succ.name]["SIMD"]
                    if ifm_ch is None or ifm_ch % mvau_simd == 0:
                        simd = mvau_simd

            config[node.name] = convinpgen_config(simd=simd)

            report["convinpgen"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "IFMChannels": ifm_ch,
                    "SIMD": simd,
                }
            )

        elif is_labelselect(node):
            pe = args.labelselect_pe
            config[node.name] = labelselect_config(pe=pe)

            report["labelselect"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "PE": pe,
                }
            )

        elif is_channelwise_or_elementwise(node):
            ch = infer_channel_count(model, node)
            pe = choose_divisor(ch, args.other_pe)

            config[node.name] = generic_pe_config(pe=pe)

            report["generic_pe"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "channels": ch,
                    "PE": pe,
                }
            )

    if len(config) == 1:
        raise RuntimeError(
            "No LeNet hardware nodes were configured. "
            "Run this on the ONNX graph after build_hw_graph / specialization, "
            "for example 13b_lenet_specialized_remaining_hw_layers.onnx."
        )

    return config, report


def print_report(report: dict):
    print("\nConfigured MVAU nodes:")
    for item in report["mvau"]:
        print(
            f"  {item['name']} ({item['op_type']}): "
            f"MW={item['MW']}, MH={item['MH']}, "
            f"SIMD={item['SIMD']}, PE={item['PE']}, "
            f"approx_cycles={item['approx_cycles']}"
        )

    print("\nConfigured Thresholding nodes:")
    for item in report["thresholding"]:
        print(
            f"  {item['name']} ({item['op_type']}): "
            f"channels={item['channels']}, PE={item['PE']}"
        )

    print("\nConfigured ConvolutionInputGenerator nodes:")
    for item in report["convinpgen"]:
        print(
            f"  {item['name']} ({item['op_type']}): "
            f"IFMChannels={item['IFMChannels']}, SIMD={item['SIMD']}"
        )

    print("\nConfigured LabelSelect nodes:")
    for item in report["labelselect"]:
        print(f"  {item['name']} ({item['op_type']}): PE={item['PE']}")

    print("\nConfigured other PE-based nodes:")
    for item in report["generic_pe"]:
        print(
            f"  {item['name']} ({item['op_type']}): "
            f"channels={item['channels']}, PE={item['PE']}"
        )

    if report["skipped"]:
        print("\nSkipped nodes:")
        for name, op_type, reason in report["skipped"]:
            print(f"  {name} ({op_type}): {reason}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LeNet-5-specific FINN/Brainsmith folding configuration JSON."
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Post-HW/pre-parallelization ONNX model, e.g. "
            "results/root/13b_lenet_specialized_remaining_hw_layers.onnx "
            "or results/root/intermediate_models/<N>_build_hw_graph.onnx."
        ),
    )

    parser.add_argument(
        "--output",
        default="configs/folding_config_test.json",
        help="Output folding config JSON path.",
    )

    parser.add_argument(
        "--pe",
        type=int,
        default=4,
        help="Preferred PE for MVAU nodes. The script will choose a valid divisor of MH <= this value.",
    )

    parser.add_argument(
        "--simd",
        type=int,
        default=4,
        help="Preferred SIMD for MVAU nodes. The script will choose a valid divisor of MW <= this value.",
    )

    parser.add_argument(
        "--threshold-pe",
        type=int,
        default=4,
        help="Preferred PE for Thresholding nodes.",
    )

    parser.add_argument(
        "--convinpgen-simd",
        type=int,
        default=4,
        help="Preferred SIMD for ConvolutionInputGenerator nodes.",
    )

    parser.add_argument(
        "--other-pe",
        type=int,
        default=4,
        help="Preferred PE for Channelwise/Elementwise/DuplicateStreams/AddStreams nodes.",
    )

    parser.add_argument(
        "--labelselect-pe",
        type=int,
        default=1,
        help="Preferred PE for LabelSelect nodes.",
    )

    parser.add_argument(
        "--runtime-writeable-weights",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether MVAU weights should be runtime writeable.",
    )

    args = parser.parse_args()

    config, report = generate_config(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nWrote LeNet folding config to: {output_path}")
    print(f"Total configured nodes: {len(config) - 1}")

    print_report(report)


if __name__ == "__main__":
    main()
