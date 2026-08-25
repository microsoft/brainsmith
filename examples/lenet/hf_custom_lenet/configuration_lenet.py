#
#  Copyright (c) 2025
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from transformers import PretrainedConfig
from dataclasses import dataclass, field


class LeNetConfig(PretrainedConfig):
    """The configuration class of LeNet model."""
    model_type = 'lenet'

    def __init__(self, activation: str = 'relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
