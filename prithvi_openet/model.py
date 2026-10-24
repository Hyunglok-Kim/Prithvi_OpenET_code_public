"""Prithvi-OpenET network and model loading.

encoder (Prithvi-EO-2.0-300M-TL, 14 x 14 x 1024 tokens) -> head (1024 -> 256 -> 32, bilinear to 224 x 224)
-> concatenated with the input channels -> refinement U-Net -> five ET/PET ratios, one per OpenET member.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import build_encoder, encode, load_pinned_weights
from .unet import RefineUNet

MEMBERS = ["disalexi", "eemetric", "geesebal", "ptjpl", "ssebop"]


class Head(nn.Module):
    """1024 -> 256 -> 32 on the token grid, then bilinear to out_size."""

    def __init__(self, in_ch=1024, mid_ch=256, out_ch=32, out_size=224):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(mid_ch, out_ch, 3, padding=1), nn.GELU())
        self.out_size = out_size

    def forward(self, feat):
        return F.interpolate(self.net(feat), size=(self.out_size, self.out_size), mode="bilinear",
                             align_corners=False)


class PrithviOpenET(nn.Module):
    """forward(x_enc, x_all, latlon, time): x_enc (B, 6, 1, H, W) normalised HLS, x_all (B, C, H, W) all normalised
    input channels, latlon (B, 2) in degrees, time (B, 2) = (year, day of year). Returns the five ET/PET ratios
    (B, 5, H, W)."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.encoder = build_encoder()
        self.head = Head()
        self.refine = RefineUNet(in_ch=32 + n_channels, base_ch=32, n_out=len(MEMBERS))

    def forward(self, x_enc, x_all, latlon, time):
        fused = self.head(encode(self.encoder, x_enc, latlon, time))
        return self.refine(torch.cat([fused, x_all], dim=1))


def load_model(weights_dir) -> tuple[PrithviOpenET, dict]:
    """Model and config.json of a released model folder (an unzipped Zenodo model zip).

    Prithvi-OpenET files hold the head and the U-Net only; their encoder is Prithvi-EO-2.0-300M-TL, unchanged, loaded
    from the pinned Hugging Face revision. Prithvi-OpenET-Tower files also hold their fine-tuned encoder."""
    from safetensors.torch import load_file

    weights_dir = Path(weights_dir)
    config = json.loads((weights_dir / "config.json").read_text(encoding="utf-8"))
    model = PrithviOpenET(len(config["raster_channels"]))
    sd = load_file(str(weights_dir / "model.safetensors"))
    has_encoder = any(k.startswith("encoder.") for k in sd)
    if not has_encoder:
        load_pinned_weights(model.encoder)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if not has_encoder:
        missing = [k for k in missing if not k.startswith("encoder.")]
    if missing or unexpected:
        raise KeyError(f"{weights_dir}: missing {list(missing)[:10]} ({len(missing)}), "
                       f"unexpected {list(unexpected)[:10]} ({len(unexpected)})")
    return model.eval(), config
