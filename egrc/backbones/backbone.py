import torch
import torch.nn as nn


class BackBone(nn.Module):
    """ViT-S token backbone used by CPEA-style few-shot heads."""

    def __init__(self, model_type="small"):
        super().__init__()
        if model_type != "small":
            raise ValueError(f"Unsupported model_type: {model_type}")
        from .vit import vit_small

        self.encoder = vit_small()

    def forward(self, images):
        return self.encoder(images)

    def forward_chunked(self, images, chunk_size=16):
        outputs = []
        for start in range(0, images.size(0), chunk_size):
            outputs.append(self.encoder(images[start : start + chunk_size]))
        return torch.cat(outputs, dim=0)
