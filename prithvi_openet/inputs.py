"""Model inputs: normalisation of the stored channel values and prediction.

Each model lists its input channels, in order, under `raster_channels` in its config.json (Opt.: 6, Opt.+F: 9,
Opt.+F+E: 14) and their statistics under `normalization`. The rules are those of training:
  HLS bands        value / divisor, 0 where the month has no HLS observation, then z-score
  awc              value / divisor, fill pixels set to AWC_IMPUTE_VALUE (awc_valid is 0 there), then z-score
  awc_valid        0/1, not normalised
  pet, vpd         one MERRA-2 value per chip-month, broadcast over the chip, then z-score
  other channels   value / divisor, fill pixels set to the channel mean, then z-score
"""
from __future__ import annotations

import json

import numpy as np
import torch

HLS_BANDS = ["hls_b02", "hls_b03", "hls_b04", "hls_nir", "hls_swir1", "hls_swir2"]   # encoder input order
AWC_IMPUTE_VALUE = 134.3   # mm


def normalize(x_int16, names, hls_valid, pet_mm: float, vpd_kpa: float, config: dict) -> np.ndarray:
    """(C, H, W) float32 model input from stored integer channels.

    x_int16 (K, H, W) stored values of the channels `names` (any order, a superset of the model's raster channels
    other than pet and vpd); hls_valid (H, W) True where the month has at least one HLS observation."""
    x = np.asarray(x_int16, dtype=np.float32)
    hls_valid = np.asarray(hls_valid, bool)
    names = list(names)
    stats = config["normalization"]["channels"]
    scalars = {"pet": float(np.float32(pet_mm)), "vpd": float(np.float32(vpd_kpa))}
    h, w = x.shape[-2:]
    out = np.empty((len(config["raster_channels"]), h, w), dtype=np.float32)
    for k, ch in enumerate(config["raster_channels"]):
        name = ch["name"]
        if ch["loader_source"] == "merra_table":
            v = np.full((h, w), scalars[name], dtype=np.float32)
        else:
            raw = x[names.index(name)]
            if ch["loader_source"] == "hls":
                v = np.where(hls_valid, raw / float(ch["stored_divisor"]), 0.0).astype(np.float32)
            else:
                invalid = (raw == float(ch["fill"])) if ch.get("fill") is not None else np.zeros_like(raw, bool)
                v = raw / float(ch["stored_divisor"])
                if name == "awc":
                    v = np.where(invalid, AWC_IMPUTE_VALUE, v)
                elif ch["normalize"] != "none" and invalid.any():
                    v = np.where(invalid, stats[name]["mean"], v)
        if ch["normalize"] != "none":
            v = (v - stats[name]["mean"]) / stats[name]["std"]
        out[k] = v
    return out


def example_inputs(ex, config: dict) -> dict:
    """Inputs of an example chip (np.load of examples/*.npz) for the model described by `config`."""
    md = json.loads(str(ex["metadata"]))
    x_all = normalize(ex["x_int16"], [str(n) for n in ex["x_int16_names"]], ex["hls_valid"],
                      float(ex["pet_mm"]), float(ex["vpd_kpa"]), config)
    names = [c["name"] for c in config["raster_channels"]]
    return {"x_all": x_all, "x_enc": x_all[[names.index(b) for b in HLS_BANDS]],
            "lat": md["lat"], "lon": md["lon"], "year": md["year"], "doy": md["doy"],
            "pet_mm": float(ex["pet_mm"]), "valid_mask": np.asarray(ex["valid_mask"], bool), "metadata": md}


def predict(model, inputs: dict) -> dict:
    """ET in mm month-1: the five predicted ET/PET ratios times the chip-month PET, and their mean."""
    with torch.no_grad():
        x_all = torch.from_numpy(np.ascontiguousarray(inputs["x_all"], dtype=np.float32))[None]
        x_enc = torch.from_numpy(np.ascontiguousarray(inputs["x_enc"], dtype=np.float32))[None, :, None]
        latlon = torch.tensor([[inputs["lat"], inputs["lon"]]], dtype=torch.float32)
        time = torch.tensor([[inputs["year"], inputs["doy"]]], dtype=torch.float32)
        ratio = model(x_enc, x_all, latlon, time)[0].float().cpu().numpy()
    et_members = ratio * float(inputs["pet_mm"])
    return {"ratio": ratio, "et_members": et_members, "et_mean": et_members.mean(axis=0)}
