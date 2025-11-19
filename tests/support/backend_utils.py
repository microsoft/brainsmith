"""Backend specialization utilities for test infrastructure.

This module provides utilities for transforming base HWCustomOp kernels
into backend-specialized variants (HLS or RTL) that support code generation
and hardware simulation.

Pattern validated by: tests/spike_backend_specialization.py
"""


from finn.util.basic import getHWCustomOp
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.primitives.transforms.specialize_kernels import SpecializeKernels
from brainsmith.registry import get_component_metadata


class MinimalBackendConfig:
    """Minimal config for SpecializeKernels in test context.

    Satisfies SpecializeKernels.__init__ requirements:
    - cfg.kernel_selections: List[Tuple[str, List[str]]]
    - cfg._resolve_fpga_part(): Returns fpgapart string
    """

    def __init__(self, fpgapart: str, kernel_selections: list[tuple[str, list[str]]]):
        self.fpgapart = fpgapart
        self.kernel_selections = kernel_selections

    def _resolve_fpga_part(self) -> str:
        """Return configured FPGA part."""
        return self.fpgapart


def specialize_to_backend(
    op,  # HWCustomOp (avoid import to prevent circular deps)
    model: ModelWrapper,
    fpgapart: str,
    backend_variants: list[type],
) -> tuple:
    """Specialize base kernel to backend variant with code generation capability.

    This function transforms a base HWCustomOp kernel (e.g., ElementwiseBinaryOp) into
    a backend-specialized variant (e.g., ElementwiseBinaryOp_hls) that inherits from
    HLSBackend or RTLBackend, enabling cppsim/rtlsim execution.

    Transformation flow:
        Stage 2: ElementwiseBinaryOp (base kernel)
                    ↓ SpecializeKernels (registry-based)
        Stage 3: ElementwiseBinaryOp_hls (backend with HLSBackend inheritance)

    This uses Brainsmith's SpecializeKernels transform which:
    - Takes kernel_selections config (kernel → backends mapping)
    - Uses registry metadata for domain determination (maintains "brainsmith.kernels")
    - Mutates op_type (adds _hls/_rtl suffix) like FINN's transform
    - Supports priority ordering (try RTL first, fallback to HLS)

    Args:
        op: Base HWCustomOp kernel instance (e.g., ElementwiseBinaryOp).
            Must have a valid op_type that has a registered backend variant.
        model: ModelWrapper containing the ONNX model with the base kernel.
        fpgapart: FPGA part string for specialization (e.g., "xc7z020clg400-1").
            This determines which backend implementations are available.
        backend_variants: List of backend classes in priority order.
            Example: [ElementwiseBinaryOp_hls]
            Example: [MVAU_rtl, MVAU_hls]  # Try RTL first, fallback to HLS

    Returns:
        Tuple[HWCustomOp, ModelWrapper]: Specialized operator and transformed model.
            - The operator will have HLSBackend or RTLBackend inheritance
            - The model will contain the specialized node with mutated op_type
            - Configuration (PE, folding, etc.) is preserved
            - Domain is correctly set to "brainsmith.kernels" for Brainsmith backends

    Raises:
        RuntimeError: If specialized node cannot be found after transformation.
            This usually means:
            - No backend could satisfy constraints for the given FPGA part
            - Backend not properly registered in Brainsmith registry

    Example:
        >>> # Import backend class
        >>> from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp_hls
        >>>
        >>> # Specialize to backend
        >>> backend_op, backend_model = specialize_to_backend(
        ...     op, model,
        ...     fpgapart="xc7z020clg400-1",
        ...     backend_variants=[ElementwiseBinaryOp_hls]
        ... )
        >>> assert backend_op.onnx_node.op_type == "ElementwiseBinaryOp_hls"
        >>> assert backend_op.onnx_node.domain == "brainsmith.kernels"
        >>> assert isinstance(backend_op, HLSBackend)
        >>> assert backend_op.get_nodeattr("PE") == 4  # Config preserved

    Example (priority order):
        >>> # Try RTL first, fallback to HLS
        >>> from brainsmith.kernels.mvau import MVAU_rtl, MVAU_hls
        >>> backend_op, backend_model = specialize_to_backend(
        ...     op, model,
        ...     fpgapart="xc7z020clg400-1",
        ...     backend_variants=[MVAU_rtl, MVAU_hls]
        ... )

    See Also:
        - brainsmith/primitives/transforms/specialize_kernels.py
        - brainsmith/registry/_lookup.py:get_domain_for_backend()
        - tests/spike_backend_specialization.py: Spike test validating this pattern
    """
    kernel_class = type(op)
    base_op_type = op.onnx_node.op_type

    # Build kernel_selections from backend_variants
    # Need to convert backend classes → fully-qualified names for registry lookup
    backend_names = []
    for backend_cls in backend_variants:
        # Get backend class name (e.g., "ElementwiseBinaryOp_hls")
        backend_class_name = backend_cls.__name__

        # Special handling for FINN ElementwiseBinary backends
        # FINN backends are not registered in Brainsmith registry
        # They use FINN's own registration system
        if (
            hasattr(backend_cls, "__module__")
            and "finn.custom_op.fpgadataflow.hls.elementwise_binary" in backend_cls.__module__
        ):
            # FINN backend - use direct class name without registry lookup
            # This will be handled by FINN's ConvertToHWLayers transform
            backend_names.append(backend_class_name)
            continue

        # Use __registry_name__ if available (handles custom names from @backend decorator)
        # Registry attaches this attribute for O(1 reverse lookup
        if hasattr(backend_cls, "__registry_name__"):
            backend_names.append(backend_cls.__registry_name__)
            continue

        # Fallback: Try to find backend in registry using class name
        # This handles backends not yet registered or loaded dynamically
        found = False
        for source in ["brainsmith", "finn", "project"]:
            candidate_name = f"{source}:{backend_class_name}"
            try:
                get_component_metadata(candidate_name, "backend")
                backend_names.append(candidate_name)
                found = True
                break
            except KeyError:
                continue

        if not found:
            raise RuntimeError(
                f"Backend class {backend_class_name} not found in registry. "
                f"Tried sources: brainsmith, finn, project. "
                f"Ensure backend is registered with @backend decorator."
            )

    # Determine kernel name (try to find in registry)
    # Base op_type might be short name without source prefix
    kernel_name = None
    for source in ["brainsmith", "finn", "project"]:
        candidate_name = f"{source}:{base_op_type}"
        try:
            get_component_metadata(candidate_name, "kernel")
            kernel_name = candidate_name
            break
        except KeyError:
            continue

    if kernel_name is None:
        # Fallback: use base_op_type directly (may work for some cases)
        kernel_name = base_op_type

    # Check if all backends are FINN backends (special handling needed)
    all_finn_backends = all(
        hasattr(bcls, "__module__") and "finn.custom_op.fpgadataflow" in bcls.__module__
        for bcls in backend_variants
    )

    if not all_finn_backends:
        # Brainsmith backends: Use SpecializeKernels transform
        # Build kernel_selections: [(kernel_name, [backend1, backend2, ...])]
        kernel_selections = [(kernel_name, backend_names)]

        # Create minimal config for SpecializeKernels
        cfg = MinimalBackendConfig(fpgapart, kernel_selections)

        # Apply Brainsmith's SpecializeKernels transform
        # This will mutate op_type and set correct domain via registry
        model = model.transform(SpecializeKernels(cfg))
    else:
        # FINN backends: Node is already specialized by FINN's InferElementwiseBinaryOperation
        # No need to run SpecializeKernels - FINN nodes already have correct op_type and domain
        pass

    # Find specialized node (SpecializeKernels mutates op_type)
    # Try each backend variant to find which one was selected
    specialized_node = None

    if all_finn_backends:
        # FINN backends: Node op_type stays as base name (e.g., "ElementwiseAdd" not "ElementwiseAdd_hls")
        # FINN nodes have domain "finn.custom_op.fpgadataflow"
        for node in model.graph.node:
            if node.op_type == base_op_type and "finn" in node.domain:
                specialized_node = node
                break
    else:
        # Brainsmith backends: op_type is mutated to backend class name
        for backend_cls in backend_variants:
            # Get expected op_type from backend class name
            expected_op_type = backend_cls.__name__  # e.g., "ElementwiseBinaryOp_hls"

            for node in model.graph.node:
                if node.op_type == expected_op_type:
                    specialized_node = node
                    break

            if specialized_node:
                break

    # Raise helpful error if node not found
    if specialized_node is None:
        available_nodes = [(n.name, n.op_type, n.domain) for n in model.graph.node]
        variant_names = [v.__name__ for v in backend_variants]
        raise RuntimeError(
            f"Failed to find specialized node after SpecializeKernels transform.\n"
            f"Base kernel: {kernel_class.__name__} (op_type={base_op_type})\n"
            f"FPGA part: {fpgapart}\n"
            f"Tried backend variants: {variant_names}\n"
            f"Registry backend names: {backend_names}\n"
            f"Available nodes: {available_nodes}\n\n"
            f"Possible causes:\n"
            f"  - No backend met constraints for FPGA part '{fpgapart}'\n"
            f"  - Backend not registered in registry (check @backend decorator)\n"
            f"  - Backend domain/op_type mismatch in registry"
        )

    # Get specialized operator instance
    if all_finn_backends:
        # FINN backends: Manual bridge - directly instantiate the backend class
        # getHWCustomOp would look up by op_type and return base class (e.g., ElementwiseAdd)
        # but we need the backend class (e.g., ElementwiseAdd_hls) for cppsim/rtlsim
        backend_cls = backend_variants[0]  # Single backend variant for FINN
        # FINN's __init__ signature: __init__(self, onnx_node, **kwargs)
        specialized_op = backend_cls(specialized_node)
    else:
        # Brainsmith backends: Use registry lookup via getHWCustomOp
        specialized_op = getHWCustomOp(specialized_node, model)

    return specialized_op, model


def verify_backend_inheritance(op, backend: str = "hls") -> bool:
    """Verify that operator has correct backend inheritance.

    Args:
        op: HWCustomOp instance to check.
        backend: Expected backend type ("hls" or "rtl").

    Returns:
        bool: True if operator has correct backend inheritance.

    Example:
        >>> op, model = specialize_to_backend(base_op, model, "xc7z020clg400-1")
        >>> assert verify_backend_inheritance(op, "hls")
    """
    if backend == "hls":
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

        return isinstance(op, HLSBackend)
    elif backend == "rtl":
        from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

        return isinstance(op, RTLBackend)
    else:
        raise ValueError(f"Unknown backend type: {backend}")


def get_backend_suffix(backend: str) -> str:
    """Get op_type suffix for backend variant.

    Args:
        backend: Backend type ("hls" or "rtl").

    Returns:
        str: Suffix for backend variant ("_hls" or "_rtl").

    Example:
        >>> get_backend_suffix("hls")
        '_hls'
    """
    if backend not in ("hls", "rtl"):
        raise ValueError(f"Unknown backend type: {backend}. Must be 'hls' or 'rtl'.")
    return f"_{backend}"
