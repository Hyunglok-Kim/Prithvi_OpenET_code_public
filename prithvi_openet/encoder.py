"""Prithvi-EO-2.0-300M-TL encoder (terratorch): construction, pinned weights and token extraction."""
from __future__ import annotations

import hashlib

import torch

# The encoder of the Prithvi-OpenET models is this file, unchanged; the Prithvi-OpenET-Tower files carry their own
# fine-tuned copy.
PRITHVI_TL = {"repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
              "revision": "63adbd39c271da4c42f447e69b1a7c91a338cdc9",
              "filename": "Prithvi_EO_V2_300M_TL.pt",
              "sha256": "3629cedfbb350faafcb0dac902ae0d3c927e25ce8d9e0024aa1276ec66956ddb"}


def _keep_token_order(x, mask_ratio):
    """MAE masking with nothing masked: every token kept, in order.

    terratorch's random_masking shuffles the tokens even at mask ratio 0, which would permute the 14 x 14 feature
    grid on every call. The models were trained and are run with this replacement."""
    n, length, _ = x.shape
    mask = torch.zeros(n, length, device=x.device, dtype=x.dtype)
    ids_restore = torch.arange(length, device=x.device).unsqueeze(0).expand(n, -1)
    return x, mask, ids_restore


def build_encoder():
    """Prithvi-EO-2.0-300M-TL architecture (ViT-L, patch 16, one frame, time and location embeddings) with
    uninitialised weights; the weights come from model.safetensors or from load_pinned_weights."""
    from terratorch.datasets.utils import HLSBands
    from terratorch.models.backbones import prithvi_vit

    encoder = prithvi_vit.prithvi_vit_300(
        pretrained=False,
        bands=[HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED, HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2],
        num_frames=1,
        coords_encoding=["time", "location"],
        coords_scale_learn=True,
    )
    encoder.random_masking = _keep_token_order
    encoder.patch_embed.img_size = (1,)   # forward_encoder reads the number of frames from img_size[0]
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.eval()


def load_pinned_weights(encoder) -> None:
    """Load Prithvi-EO-2.0-300M-TL from the pinned Hugging Face revision (about 1.3 GB, downloaded once into the
    Hugging Face cache, SHA-256 checked). The MAE decoder and the positional embedding of the checkpoint are not
    loaded: the decoder is not used and the positional embedding is computed in the forward pass."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=PRITHVI_TL["repo_id"], filename=PRITHVI_TL["filename"],
                           revision=PRITHVI_TL["revision"])
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != PRITHVI_TL["sha256"]:
        raise RuntimeError(f"{path}: sha256 {h.hexdigest()} is not the pinned Prithvi-EO-2.0-300M-TL file")
    sd = torch.load(path, map_location="cpu")
    sd = {k.removeprefix("encoder."): v for k, v in sd.items()
          if not (k.startswith("decoder.") or ".decoder." in k) and "pos_embed" not in k}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not (k.startswith("decoder") or k == "mask_token")]
    if missing or unexpected:
        raise KeyError(f"Prithvi-EO-2.0-300M-TL checkpoint: missing {missing[:10]}, unexpected {unexpected[:10]}")


def encode(encoder, x_enc, latlon, time):
    """x_enc (B, 6, 1, H, W) normalised HLS, latlon (B, 2) in degrees, time (B, 2) = (year, day of year).
    Returns the token grid as a feature map (B, 1024, H/16, W/16)."""
    tokens = encoder.forward_encoder(x_enc, time.float().unsqueeze(1), latlon.float())[0]
    tokens = tokens[:, 1:, :]                                      # drop the CLS token
    b, n, d = tokens.shape
    g = int(round(n ** 0.5))
    return tokens.permute(0, 2, 1).contiguous().view(b, d, g, g)
