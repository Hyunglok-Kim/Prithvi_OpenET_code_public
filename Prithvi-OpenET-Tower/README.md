# Prithvi-OpenET-Tower

Prithvi-OpenET adapted to flux-tower evapotranspiration. Starting from the Prithvi-OpenET model of the same input
configuration, the encoder is unfrozen and the model is trained on OpenET patches and flux-tower ET together, so it
keeps the OpenET spatial pattern while matching tower ET.

## Models

| Model | Inputs | Zenodo file |
|---|---|---|
| Prithvi-OpenET-Tower (Opt.) | 6 HLS bands | `prithvi_openet_tower_opt.zip` |
| Prithvi-OpenET-Tower (Opt.+F) | + shortwave radiation, PET, VPD | `prithvi_openet_tower_opt_f.zip` |
| Prithvi-OpenET-Tower (Opt.+F+E) | + elevation, TWI, AWC and mask, aridity | `prithvi_openet_tower_opt_f_e.zip` |

## How the models were trained

Adapted from Prithvi-OpenET with a joint loss (OpenET pattern + 0.2 × OpenET level + 1.0 × tower) on closure-corrected
FLUXNET-1F tower ET over the ETo-weighted flux source area, 15 epochs with the encoder unfrozen, and the checkpoint chosen on
validation tower clusters among epochs that keep the OpenET pattern error close to that of Prithvi-OpenET.

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
python scripts/run_example.py --weights weights/prithvi_openet_tower_opt_f_e --example examples/conus_T13TGE_2020-07.npz --out example_output --selftest
```

`--selftest` checks the prediction against the one stored in the chip file.
