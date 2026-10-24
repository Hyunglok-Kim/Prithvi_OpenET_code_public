# Prithvi-OpenET

An emulator of the OpenET Collection v2.1 monthly ensemble at 30 m. The Prithvi-EO-2.0-300M-TL encoder is frozen; the
convolutional head, the refinement U-Net and its five heads (one per OpenET member) are trained on OpenET labels.

## Models

| Model | Inputs | Zenodo file |
|---|---|---|
| Prithvi-OpenET (Opt.) | 6 HLS bands | `prithvi_openet_opt.zip` |
| Prithvi-OpenET (Opt.+F) | + shortwave radiation, PET, VPD | `prithvi_openet_opt_f.zip` |
| Prithvi-OpenET (Opt.+F+E) | + elevation, TWI, AWC and mask, aridity | `prithvi_openet_opt_f_e.zip` |

## How the models were trained

Trained on OpenET v2.1 monthly ET (2016-2024) of 34 CONUS MGRS tiles: 224 × 224 patches, masked Huber loss in ET units
averaged over the five heads, AdamW (lr 4.6e-5), 30 epochs, with the checkpoint chosen on an internal validation set of
10 % of patch locations. The frozen encoder is Prithvi-EO-2.0-300M-TL, unchanged; the released
files hold only the head and the U-Net, and the encoder is loaded from Hugging Face at a pinned revision on first use.

## Example

`examples/example_inference.ipynb` runs the three models on an example chip and shows the inputs, the ET maps and the
five OpenET-member heads. The chips are in `../examples/` (shared by both model families):

| Chip | Location | Month |
|---|---|---|
| `conus_T13TGE_2020-07` | MGRS T13TGE, eastern Colorado (centre-pivot irrigation) | July 2020 |
| `conus_T15SXU_2021-08` | MGRS T15SXU, irrigated cropland | August 2021 |
| `france_lamasquere_2021-07` | flux tower FR-Lam, Lamasquère, France (MGRS T31TCJ) | July 2021 |

From the command line (after unzipping the model zips from Zenodo, to be released, into `weights/` in the repository root):

```bash
python scripts/run_example.py --weights weights/prithvi_openet_opt_f_e --example examples/conus_T13TGE_2020-07.npz --out example_output --selftest
```

`--selftest` checks the prediction against the one stored in the chip file.
