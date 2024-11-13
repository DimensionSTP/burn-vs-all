import torch
from torch import nn

import timm


class TimmModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        pretrained: str,
        num_labels: int,
        classification_type: int,
    ) -> None:
        super().__init__()
        if pretrained == "pretrained":
            is_pretrained = True
        elif pretrained == "raw":
            is_pretrained = False
        else:
            raise ValueError(f"Invalid pretrained: {pretrained}")
        self.model = timm.create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=num_labels if classification_type not in range(4) else 2,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(image)
        return output
