# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SpecializeKernels: Registry-based backend specialization transform.

Replaces generic kernel nodes with specialized backend nodes based on kernel_selections
configuration. Uses registry metadata for source-to-domain mapping.

Architecture:
- kernel_selections provides priority lists: [(kernel_name, [backend1, backend2, ...])]
- Try backends left-to-right, select first that passes constraint checks
- Replace kernel node with backend node (mutate op_type, domain, backend attribute)
- Domain determined by backend's source via get_domain_for_backend()
"""

import logging
import warnings

import numpy as np
from finn.util.basic import get_dsp_block, getHWCustomOp, is_versal
from onnx import helper
from qonnx.transformation.base import Transformation

from brainsmith.registry import get_component_metadata, get_domain_for_backend

logger = logging.getLogger(__name__)


# ============================================================================
# Constraint Checking Functions (ported from FINN)
# ============================================================================


def _dwc_rtl_possible(node, model=None):
    """Check if StreamingDataWidthConverter can use RTL variant.

    RTL variant requires integer width ratios (one width divides the other).
    """
    dwc = getHWCustomOp(node, model)
    dwc_in_width = dwc.get_nodeattr("inWidth")
    dwc_out_width = dwc.get_nodeattr("outWidth")
    # Check if rtl variant can be used
    iwidth_d = dwc_in_width % dwc_out_width == 0
    owidth_d = dwc_out_width % dwc_in_width == 0
    return iwidth_d or owidth_d


def _mvu_rtl_possible(node, fpgapart, model):
    """Check whether RTL-based MVAU is supported.

    Constraints:
    - No embedded thresholding (noActivation == 1)
    - No binaryXnorMode (binaryXnorMode == 0)
    - Signed weights required
    - DSP48E1: narrow weights only
    - Bitwidth limits: 2-8 bits for weights, 2-8 (or 9-bit signed) for inputs
    """
    node_inst = getHWCustomOp(node, model)

    # Check for embedded thresholding and binary xnor mode
    no_activation = node_inst.get_nodeattr("noActivation") == 0
    not_binaryxnor_mode = node_inst.get_nodeattr("binaryXnorMode") == 1
    if no_activation or not_binaryxnor_mode:
        return False

    # Check if weights are signed
    wdt = node_inst.get_input_datatype(1)

    # Check which DSP block is available on FPGA
    dsp_block = get_dsp_block(fpgapart)

    # Check if weights are narrow
    weights = model.get_initializer(node.input[1])
    if weights is None:
        weights_min = wdt.min()
    else:
        weights_min = np.min(weights)
    narrow_weights = False if weights_min == wdt.min() else True

    # If non-narrow weights and only DSP48E1 available, RTL not possible
    if not narrow_weights and dsp_block == "DSP48E1":
        return False

    # Check if input and weight data types are in range
    idt = node_inst.get_input_datatype()
    inp_width_in_range = (2 <= idt.bitwidth() <= 8) or (idt.bitwidth() == 9 and idt.signed())
    weight_width_in_range = 2 <= wdt.bitwidth() <= 8

    return inp_width_in_range and weight_width_in_range


def _vvu_rtl_possible(node, fpgapart, model=None):
    """Check whether RTL-based VVU is supported.

    Constraints:
    - Versal-only (DSP58)
    - No embedded thresholding (noActivation == 1)
    - Signed weights required
    - Bitwidth limits: ≤8 bits for weights, ≤8 (or 9-bit signed) for inputs
    """
    node_inst = getHWCustomOp(node, model)

    # Check for embedded thresholding
    if not node_inst.get_nodeattr("noActivation"):
        return False

    # Versal-only
    if not is_versal(fpgapart):
        return False

    # Check bitwidth constraints
    idt = node_inst.get_input_datatype(0)
    wdt = node_inst.get_input_datatype(1)
    in_width_in_range = (idt.bitwidth() <= 8) or (idt.bitwidth() == 9 and idt.min() < 0)
    weight_width_in_range = wdt.bitwidth() <= 8
    signed_weights = wdt.min() < 0

    return in_width_in_range and weight_width_in_range and signed_weights


# ============================================================================
# SpecializeKernels Transform
# ============================================================================


class SpecializeKernels(Transformation):
    """Specialize hardware kernel nodes to backend implementations.

    Uses kernel_selections config to determine backend priorities, validates
    constraints, and replaces nodes with specialized backends.

    Backend selection:
    - Try backends in priority order (left-to-right in list)
    - First backend passing constraint checks is selected
    - Falls back to warnings if no viable backend found

    Node transformation:
    - op_type: kernel name → backend class name (e.g., "MVAU" → "MVAU_hls")
    - domain: from backend's source via get_domain_for_backend()
    - backend: language attribute ("hls" or "rtl")
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fpgapart = cfg._resolve_fpga_part()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        # Build backend map from kernel_selections
        # Format: {kernel_name: [backend1, backend2, ...]}
        backend_map = self._build_backend_map(model)

        for node in graph.node:
            # Skip nodes that are not hardware layers
            if not self._is_hw_node(node):
                node_ind += 1
                continue

            # Get backend priority list for this kernel
            backend_list = backend_map.get(node.op_type, [])
            if not backend_list:
                # No backends configured for this kernel, skip
                node_ind += 1
                continue

            # Try backends in priority order
            selected_backend = self._select_viable_backend(node, backend_list, model)

            if selected_backend is None:
                warnings.warn(
                    f"No viable backend found for node {node.name} ({node.op_type}). "
                    f"Tried: {backend_list}. Skipping specialization."
                )
                node_ind += 1
                continue

            # Create specialized node
            new_node = self._create_specialized_node(node, selected_backend)

            # Replace node in graph
            graph.node.insert(node_ind, new_node)
            graph.node.remove(node)
            graph_modified = True

            node_ind += 1

        return (model, graph_modified)

    def _build_backend_map(self, model):
        """Build map of kernel name to backend priority list.

        Extracts backend lists from kernel_selections attribute on cfg.
        Format: [("source:kernel", ["source:backend1", "source:backend2", ...])]

        Returns:
            Dict[str, List[str]]: {short_kernel_name: [backend_name1, backend_name2, ...]}
        """
        kernel_selections = getattr(self.cfg, "kernel_selections", None)

        if not kernel_selections:
            return {}

        backend_map = {}
        for kernel_name, backend_list in kernel_selections:
            # Strip source prefix to get short name for matching node.op_type
            # "source:KernelName" → "KernelName"
            short_name = kernel_name.split(":", 1)[1] if ":" in kernel_name else kernel_name

            # Ensure backend_list is a list (should always be list now)
            if not isinstance(backend_list, list):
                backend_list = [backend_list]

            backend_map[short_name] = backend_list

        return backend_map

    def _is_hw_node(self, node):
        """Check if node is a hardware layer (unspecialized)."""
        return node.domain.endswith(".custom_op.fpgadataflow") or (
            node.domain.startswith("brainsmith.kernels")
            and not (node.domain.endswith(".hls") or node.domain.endswith(".rtl"))
        )

    def _select_viable_backend(self, node, backend_list, model):
        """Select first viable backend from priority list.

        Args:
            node: ONNX node to specialize
            backend_list: List of backend names in priority order
            model: ModelWrapper

        Returns:
            Backend name if viable backend found, None otherwise
        """
        logger.debug(f"Selecting backend for {node.name} ({node.op_type}), trying: {backend_list}")

        for backend_name in backend_list:
            try:
                # Get backend metadata
                meta = get_component_metadata(backend_name, "backend")
                language = meta.backend_language

                # Check if backend is viable
                if self._check_backend_viable(node, language, model):
                    logger.debug(f"Selected backend {backend_name} ({language}) for {node.name}")
                    return backend_name
                else:
                    logger.debug(f"Backend {backend_name} ({language}) not viable for {node.name}")

            except KeyError:
                warnings.warn(f"Backend not found in registry: {backend_name}")
                continue

        return None

    def _check_backend_viable(self, node, language, model):
        """Check if backend language is viable for this node.

        Applies constraint checking for RTL backends.

        Args:
            node: ONNX node
            language: Backend language ("hls" or "rtl")
            model: ModelWrapper

        Returns:
            bool: True if backend is viable
        """
        # HLS always viable (fewest constraints)
        if language == "hls":
            return True

        # RTL requires constraint checking
        optype = node.op_type

        if optype == "StreamingDataWidthConverter":
            viable = _dwc_rtl_possible(node, model)
            logger.debug(f"RTL constraint check for {node.name} (DWC): {viable}")
            return viable
        elif optype == "MVAU":
            viable = _mvu_rtl_possible(node, self.fpgapart, model)
            logger.debug(f"RTL constraint check for {node.name} (MVAU): {viable}")
            return viable
        elif optype == "VectorVectorActivation":
            viable = _vvu_rtl_possible(node, self.fpgapart, model)
            logger.debug(f"RTL constraint check for {node.name} (VVU): {viable}")
            return viable
        else:
            # For other ops, assume RTL is viable if it exists
            logger.debug(f"RTL assumed viable for {node.name} ({optype})")
            return True

    def _create_specialized_node(self, node, backend_name):
        """Create specialized backend node.

        Args:
            node: Original kernel node
            backend_name: Selected backend name (e.g., 'brainsmith:LayerNorm_hls')

        Returns:
            New ONNX node with specialized backend
        """
        # Get backend metadata
        meta = get_component_metadata(backend_name, "backend")
        language = meta.backend_language

        # Extract backend class name from full name
        # E.g., 'brainsmith:LayerNorm_hls' → 'LayerNorm_hls'
        backend_class_name = backend_name.split(":", 1)[1] if ":" in backend_name else backend_name

        # Get domain for backend
        domain = get_domain_for_backend(backend_name)

        # Create new node
        new_node = helper.make_node(
            backend_class_name,
            node.input,
            node.output,
            domain=domain,
            name=node.name,
        )

        # Copy all attributes except backend
        for attribute in node.attribute:
            if attribute.name != "backend":
                new_node.attribute.append(attribute)

        # Set backend attribute
        new_node.attribute.append(helper.make_attribute("backend", language))

        # Copy metadata_props (PyTorch hierarchy metadata for MLO loop rolling)
        for prop in node.metadata_props:
            new_node.metadata_props.append(prop)

        return new_node
