# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shell integration metadata extraction transform."""

import json
import os

import numpy as np
from finn.util.mlo_sim import dat_file_to_numpy_array
import qonnx.custom_op.registry as registry
from qonnx.transformation.base import Transformation


class ExtractShellIntegrationMetadata(Transformation):
    """Walks the ONNX graph and extracts all relevant metadata for shell integration
    handover."""

    def __init__(self, metadata_file: str):
        super().__init__()
        self.metadata_file: str = metadata_file
        self.md = {}

    def apply(self, model):
        graph = model.graph

        # destination dir to copy artifacts
        dirname = os.path.dirname(self.metadata_file)

        # Search for FINNLoop ops (Does not currently support nested FINNLoops)
        finn_loops={}
        mlo = False
        for node in model.graph.node:
            if node.op_type == "FINNLoop":
                finnloop_op = registry.getCustomOp(node)
                finnloop_body = finnloop_op.get_nodeattr("body");

                mvau_hbm_weights = {}
                extern_idx = 0
                for idx, lb_inp in enumerate(finnloop_body.graph.input):
                    downstream = finnloop_body.find_consumer(lb_inp.name)
                    if downstream.op_type.startswith("MVAU"):
                        mlo = True
                        mvau_hbm_weights[idx] = {}
                        mvau_hbm_weights[idx]["name"] = lb_inp.name
                        datfile = (
                            f"{finnloop_op.get_nodeattr('code_gen_dir_ipgen')}/memblock_MVAU_rtl_id_{idx}.dat"
                        )

                        # Save the weights as a numpy file
                        np_dat = dat_file_to_numpy_array(datfile)
                        mvau_hbm_weights[idx]["weight_npy"] = f"memblock_MVAU_rtl_id_{idx}.npy"
                        np.save(f"{dirname}/{mvau_hbm_weights[idx]['weight_npy']}", np_dat)

                        # Copy to the destination dir
                        mvau_hbm_weights[idx]["extern_idx"] = extern_idx
                        mvau_hbm_weights[idx]["extern_name"] = f"m_axi_MVAU_id_{idx}"
                        mlo_mvau = registry.getCustomOp(downstream)
                        mvau_hbm_weights[idx]["PE"] = mlo_mvau.get_nodeattr("PE")
                        mvau_hbm_weights[idx]["SIMD"] = mlo_mvau.get_nodeattr("SIMD")
                        mvau_hbm_weights[idx]["MH"] = mlo_mvau.get_nodeattr("MH")
                        mvau_hbm_weights[idx]["MW"] = mlo_mvau.get_nodeattr("MW")
                        mvau_hbm_weights[idx]["weightDataType"] = mlo_mvau.get_nodeattr("weightDataType")
                        extern_idx += 1
                finn_loops[node.name] = mvau_hbm_weights
        self.md["mlo"] = mlo
        self.md["finn_loops"] = finn_loops


        # Extract instream widths
        instreams = {}
        for input_tensor in graph.input:
            consumer = model.find_consumer(input_tensor.name)
            inst = registry.getCustomOp(consumer)
            instreams[input_tensor.name] = {
                "datatype": inst.get_input_datatype().name,
                "width": inst.get_instream_width(),
                "shape": inst.get_normal_input_shape(),
            }

        self.md['instreams'] = instreams

        outstreams = {}
        for output_tensor in graph.output:
            producer = model.find_producer(output_tensor.name)
            inst = registry.getCustomOp(producer)
            outstreams[output_tensor.name] = {
                "datatype": inst.get_output_datatype().name,
                "width": inst.get_outstream_width(),
                "shape": inst.get_normal_output_shape()
            }
        self.md["outstreams"] = outstreams

        static_matmuls = {}
        for node in graph.node:
            if node.op_type == "MVAU_rtl":
                inst = registry.getCustomOp(node)
                mm = {}
                mm["MH"] = inst.get_nodeattr("MH")
                mm["MW"] = inst.get_nodeattr("MW")
                mm["SIMD"] = inst.get_nodeattr("SIMD")
                mm["PE"] = inst.get_nodeattr("PE")
                static_matmuls[node.name] = mm
        self.md["static_matmuls"] = static_matmuls

        with open(self.metadata_file, "w") as fp:
            json.dump(self.md, fp, indent=4)

        return(model, False)
