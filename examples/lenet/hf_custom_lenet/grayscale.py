#
#  Copyright (c) 2025
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import torch
import numpy as np
from typing import Any
from transformers import BaseImageProcessorFast
from transformers.image_utils import ImageInput


class GrayScaleImageProcessorFast(BaseImageProcessorFast):
    def preprocess(self, examples: dict[str, list[Any]], **kwargs) -> dict[str, torch.Tensor]:
        pixel_values = []

        for img in examples['img']:
            img_tensor = torch.from_numpy(np.array(img)[..., 3]).float()
            pixel_values.append(img_tensor.unsqueeze(0))

        pixel_values = torch.stack(pixel_values, dim=0)
        pixel_values = pixel_values / 255.0

        labels = list(map(int, examples['label']))

        return {
            "pixel_values": pixel_values.to(device='mps'),
            "labels": torch.tensor(labels).to(device='mps')
        }
