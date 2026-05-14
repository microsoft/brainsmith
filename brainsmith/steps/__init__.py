# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Steps Module

Auto-imports step modules to trigger @step decorator registrations.
All step functionality is available through the unified plugin system:

    from brainsmith.core.plugins import get_step, list_steps
    
    # Get step function by name
    step_fn = get_step("shell_metadata_handover")
    model = step_fn(model, cfg)
    
    # List all available steps
    steps = list_steps()
"""

# Import all step modules to trigger registration
from . import core_steps
from . import bert_custom_steps
from . import kernel_inference
from . import v80_deployment_steps