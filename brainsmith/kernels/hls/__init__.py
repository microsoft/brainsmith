# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# flake8: noqa
# Disable linting from here, as all imports will be flagged E402 and maybe F401

"""
Brainsmith HLS Kernel Imports

This is a TEMPORARY measure to ensure HLS variants are properly registered
in the kernels.hls namepace until backend refactoring is complete.

Similar to how FINN imports its HLS variants in:
deps/finn/src/finn/custom_op/fpgadataflow/hls/__init__.py
"""

# Import all HLS custom ops - they will be discovered automatically via namespace
# Note: Using absolute imports to ensure proper registration

# Import Brainsmith HLS kernels
from brainsmith.kernels.crop.crop_hls import Crop_hls
