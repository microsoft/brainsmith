# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint kernel parsing.

Handles parsing of kernel specifications and backend resolution.
"""

import logging

from brainsmith.registry import get_backend, list_backends_for_kernel

logger = logging.getLogger(__name__)


def parse_kernels(kernels_data: list[str | dict]) -> list[tuple[str, list[type]]]:
    """Parse kernels section with optional backend specification.

    Supports two formats:
    1. String: Kernel name only → all registered backends
    2. Dict: Kernel name with backend list → specific backends in priority order

    Args:
        kernels_data: List of kernel specifications, each either:
            - str: Kernel name (e.g., 'LayerNorm') → uses all registered backends
            - dict: {kernel_name: [backend_names]} → uses specified backends

    Returns:
        List of (kernel_name, backend_classes) tuples where backend_classes
        are in priority order (first backend is tried first during specialization).
        Kernels with no backends will have an empty backend_classes list and will
        not be specialized during specialize_kernel_backends (a warning is logged).

    Raises:
        ValueError: If kernel spec format is invalid
        ValueError: If specified backend not found in registry
        ValueError: If specified backend doesn't match kernel

    Note:
        Kernels without backends are allowed. They will be included in the design
        space but skipped during backend specialization. This enables registration
        of kernels that don't yet have backend implementations.

    Examples:
        >>> # All backends for each kernel (auto-sorted: RTL first, then HLS)
        >>> parse_kernels(['LayerNorm', 'Crop'])
        [('LayerNorm', [<class 'LayerNorm_rtl'>, <class 'LayerNorm_hls'>]),
         ('Crop', [<class 'Crop_hls'>])]

        >>> # Specific backends in priority order (user-specified order preserved)
        >>> parse_kernels([
        ...     'LayerNorm',
        ...     {'MVAU': ['MVAU_rtl', 'MVAU_hls']},
        ...     {'Softmax': ['brainsmith:Softmax_hls']}
        ... ])
        [('LayerNorm', [<class 'LayerNorm_rtl'>, <class 'LayerNorm_hls'>]),
         ('MVAU', [<class 'MVAU_rtl'>, <class 'MVAU_hls'>]),
         ('Softmax', [<class 'Softmax_hls'>])]
    """
    kernel_backends = []

    for spec in kernels_data:
        # Parse kernel specification
        if isinstance(spec, str):
            # Format 1: String → all backends
            kernel_name = spec
            backend_classes = _resolve_all_backends(kernel_name)
        elif isinstance(spec, dict):
            # Format 2: Dict → specific backends
            if len(spec) != 1:
                raise ValueError(
                    f"Kernel dict spec must have exactly one key, got {len(spec)}: {spec}\n"
                    f"Example: {{'MVAU': ['MVAU_rtl', 'MVAU_hls']}}"
                )
            kernel_name, backend_spec = next(iter(spec.items()))
            backend_classes = _resolve_specified_backends(kernel_name, backend_spec)
        else:
            raise ValueError(
                f"Kernel spec must be a string or dict, got {type(spec).__name__}: {spec}\n"
                f"Examples:\n"
                f"  - 'LayerNorm'  # All backends\n"
                f"  - {{'MVAU': ['MVAU_rtl', 'MVAU_hls']}}  # Specific backends"
            )

        if not backend_classes:
            logger.debug(
                f"Kernel '{kernel_name}' has no registered backends. "
                f"This kernel will not be specialized during specialize_kernel_backends."
            )

        kernel_backends.append((kernel_name, backend_classes))

    return kernel_backends


def _resolve_all_backends(kernel_name: str) -> list[type]:
    """Get all registered backends for a kernel.

    Args:
        kernel_name: Name of the kernel (e.g., 'MVAU', 'LayerNorm')

    Returns:
        List of backend classes sorted by priority (RTL first, then HLS)

    Raises:
        ValueError: If backend resolution fails
    """
    backend_names = list_backends_for_kernel(kernel_name)

    if not backend_names:
        return []

    # Resolve backend names to classes
    try:
        backend_classes = [get_backend(name) for name in backend_names]
    except KeyError as e:
        raise ValueError(f"Backend not found for kernel '{kernel_name}': {e}") from e

    # Sort backends: RTL first, then HLS, then others
    def backend_sort_key(backend_class: type) -> tuple:
        """Sort key: (priority, class_name) where lower priority comes first."""
        class_name = backend_class.__name__.lower()
        if class_name.endswith("_rtl"):
            priority = 0  # RTL first
        elif class_name.endswith("_hls"):
            priority = 1  # HLS second
        else:
            priority = 2  # Others last
        return (priority, class_name)

    backend_classes.sort(key=backend_sort_key)

    return backend_classes


def _resolve_specified_backends(kernel_name: str, backend_spec: list[str]) -> list[type]:
    """Resolve explicitly specified backends for a kernel.

    Accepts both class names (e.g., 'MVAU_rtl') and full registry names
    (e.g., 'brainsmith:MVAU_rtl').

    Args:
        kernel_name: Name of the kernel (e.g., 'MVAU')
        backend_spec: List of backend names in priority order

    Returns:
        List of backend classes in the specified priority order

    Raises:
        ValueError: If backend_spec is not a list
        ValueError: If any backend name is not found
        ValueError: If any backend doesn't match the kernel
    """
    if not isinstance(backend_spec, list):
        raise ValueError(
            f"Backend specification for '{kernel_name}' must be a list, "
            f"got {type(backend_spec).__name__}: {backend_spec}\n"
            f"Example: {{'MVAU': ['MVAU_rtl', 'MVAU_hls']}}"
        )

    if not backend_spec:
        raise ValueError(
            f"Backend list for '{kernel_name}' cannot be empty. "
            f"Either specify backends or use string format to get all backends."
        )

    backend_classes = []
    for backend_name in backend_spec:
        if not isinstance(backend_name, str):
            raise ValueError(
                f"Backend name must be a string, got {type(backend_name).__name__}: {backend_name}"
            )

        # Try to resolve backend name (supports both class names and registry names)
        backend_class = _resolve_backend_name(kernel_name, backend_name)
        backend_classes.append(backend_class)

    return backend_classes


def _resolve_backend_name(kernel_name: str, backend_name: str) -> type:
    """Resolve a backend name to its class.

    Supports two formats:
    1. Class name: 'MVAU_rtl' → searches registry for backends matching this kernel
    2. Registry name: 'brainsmith:MVAU_rtl' → directly looks up in registry

    Args:
        kernel_name: Name of the kernel this backend should implement
        backend_name: Backend name (class name or registry name)

    Returns:
        Backend class

    Raises:
        ValueError: If backend not found or doesn't match kernel
    """
    # If backend_name contains ':', it's a full registry name
    if ":" in backend_name:
        try:
            backend_class = get_backend(backend_name)
        except KeyError:
            raise ValueError(
                f"Backend '{backend_name}' not found in registry for kernel '{kernel_name}'"
            )
    else:
        # Class name format - search for it among kernel's backends
        all_backend_names = list_backends_for_kernel(kernel_name)

        # Look for a backend with this class name (after the colon)
        # Use case-insensitive matching since registry names may differ in case from class names
        matching_backend = None
        for full_name in all_backend_names:
            # Extract class name from registry name (e.g., 'brainsmith:MVAU_rtl' → 'MVAU_rtl')
            class_name = full_name.split(":")[-1]
            if class_name.lower() == backend_name.lower():
                matching_backend = full_name
                break

        if matching_backend is None:
            available = [name.split(":")[-1] for name in all_backend_names]
            raise ValueError(
                f"Backend '{backend_name}' not found for kernel '{kernel_name}'.\n"
                f"Available backends for {kernel_name}: {available}"
            )

        try:
            backend_class = get_backend(matching_backend)
        except KeyError:
            raise ValueError(
                f"Backend '{backend_name}' found but could not be loaded from registry"
            )

    # Validate that backend implements this kernel
    _validate_backend_kernel_match(kernel_name, backend_name, backend_class)

    return backend_class


def _validate_backend_kernel_match(
    kernel_name: str, backend_name: str, backend_class: type
) -> None:
    """Validate that a backend implements the expected kernel.

    Args:
        kernel_name: Expected kernel name (e.g., 'MVAU' or 'finn:MVAU')
        backend_name: Backend name from YAML (for error messages)
        backend_class: Backend class to validate

    Raises:
        ValueError: If backend doesn't implement the kernel
    """
    # Strip namespace from kernel_name if present (e.g., 'finn:MVAU' → 'MVAU')
    expected_kernel = kernel_name.split(":")[-1]

    # Get the backend's target kernel from its class name
    # Backend class names follow pattern: KernelName_backend (e.g., MVAU_rtl, LayerNorm_hls)
    backend_class_name = backend_class.__name__

    # Extract kernel part (everything before last underscore for _hls/_rtl)
    if "_" in backend_class_name:
        backend_kernel = backend_class_name.rsplit("_", 1)[0]
    else:
        backend_kernel = backend_class_name

    # Allow exact match or backend being more specific (e.g., MVAU_rtl implements MVAU)
    if backend_kernel != expected_kernel:
        raise ValueError(
            f"Backend '{backend_name}' (class: {backend_class_name}) does not implement "
            f"kernel '{kernel_name}'. Expected backend for kernel '{kernel_name}', "
            f"but backend implements '{backend_kernel}'."
        )
