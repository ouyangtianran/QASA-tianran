"""Pretrained encoders used by the published QASA scripts."""

import torch


PATCH_SIZES = {
    "dino_vitb16": 16,
    "dinov2_vits14": 14,
}


def build_encoder(name, image_size):
    if name not in PATCH_SIZES:
        raise ValueError(
            f"Unsupported encoder {name!r}; choose one of {sorted(PATCH_SIZES)}"
        )

    patch_size = PATCH_SIZES[name]
    max_tokens = (image_size // patch_size) ** 2
    if name == "dino_vitb16":
        encoder = torch.hub.load("facebookresearch/dino:main", name)
    else:
        encoder = torch.hub.load("facebookresearch/dinov2", name)
    return encoder, max_tokens
