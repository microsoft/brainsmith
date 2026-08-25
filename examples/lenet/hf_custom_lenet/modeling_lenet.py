#
#  Copyright (c) 2025
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from transformers import PreTrainedModel

from .configuration_lenet import LeNetConfig

ACTIVATION_MAPPING = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "lrelu": nn.LeakyReLU
}

ACTIVATION_OPTION = Literal['relu', 'tanh', 'sigmoid', 'lrelu']


class LeNetLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ACTIVATION_OPTION = 'tanh',
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=padding,
            dilation=dilation,
            stride=1,
            bias=True
        )
        self.act = ACTIVATION_MAPPING[activation]()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, tensor: Tensor):
        x = self.conv1(tensor)
        x = self.act(x)
        x = self.pool1(x)

        return x


class LeNetPretrainedModel(PreTrainedModel):
    config_class = LeNetConfig
    main_input_name = "pixel_values"
    base_model_prefix = "lenet"
    input_modalities = ("image", )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(module.bias, -bound, bound)


class LeNetModel(LeNetPretrainedModel):
    """LeNet Model class."""

    def __init__(self, config: LeNetConfig):
        super().__init__(config)

        self.layer1 = LeNetLayer(in_channels=1, out_channels=6, padding=2)
        self.layer2 = LeNetLayer(in_channels=6, out_channels=16)
        self.conv1x1 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.activation = ACTIVATION_MAPPING[config.activation]()

        self.post_init()

    def forward(self, tensor: Tensor) -> Tensor:
        x = self.layer1(tensor)
        x = self.layer2(x)

        x = self.activation(x)
        x = self.conv1x1(x)
        x = torch.flatten(x, start_dim=1)

        return x


class LeNetForImageClassification(LeNetPretrainedModel):
    """LeNet Model for Image Classification class."""

    def __init__(self, config: LeNetConfig):
        super().__init__(config)

        self.letnet = LeNetModel(config)
        self.head = nn.Sequential(
            nn.Linear(in_features=120, out_features=84, bias=True),
            ACTIVATION_MAPPING[config.activation](),
            nn.Linear(in_features=84, out_features=10, bias=True),
        )

        self.post_init()

    def forward(self, pixel_values: Tensor, labels: list[int] | Tensor = None) -> dict | Tensor:
        x = self.letnet(pixel_values)
        logits = self.head(x)

        if labels is not None:
            loss = F.cross_entropy(input=logits, target=labels)

            return {"logits": logits, "loss": loss}

        return logits
