# Prithvi-OpenET: Geospatial Foundation Model-Based 30 m Monthly Evapotranspiration Estimation beyond the Conterminous United States with Landsat and Sentinel-2 Imagery

#### Euiyoung Choi, Seulgi Kim, Hyunglok Kim
#### HydroAI Lab, Gwangju Institute of Science and Technology

**Paper:** Prithvi-OpenET: Geospatial Foundation Model-Based 30 m Monthly Evapotranspiration Estimation beyond the Conterminous United States with Landsat and Sentinel-2 Imagery. *Remote Sensing of Environment* (submitted), doi:TODO. **Code:** https://github.com/Hyunglok-Kim/Prithvi_OpenET_code_public. **Model weights:** to be released on Zenodo.

This repository contains the inference code and examples for two families of models built on the
[Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL) encoder:

- **[Prithvi-OpenET](Prithvi-OpenET/)**: an emulator of the OpenET ensemble, trained on OpenET Collection v2.1 monthly
  ET over CONUS (stage 1).
- **[Prithvi-OpenET-Tower](Prithvi-OpenET-Tower/)**: Prithvi-OpenET adapted to global eddy-covariance flux towers while
  keeping the OpenET spatial pattern (stage 2).

Each family comes in three input configurations, so six models are released. The model weights will be made available on
Zenodo later (not in this repository).

## Model overview

One monthly HLS composite (six bands) is encoded by Prithvi-EO-2.0-300M-TL with its location (latitude, longitude)
and date (year, day of year) embeddings. A convolutional head projects the 14 × 14 tokens to 32 channels, which are
upsampled to 224 × 224 at 30 m, concatenated with the auxiliary rasters of the input configuration and refined by a
U-Net with five heads. Each head predicts the ratio of one OpenET member's ET (DisALEXI, eeMETRIC, geeSEBAL, PT-JPL,
SSEBop) to MERRA-2 FAO-56 reference ET (PET); ET = ratio × PET. Maps and tower comparisons use the five-head mean.

| Input configuration | Channels | Inputs |
|---|---|---|
| Opt. | 6 | HLS B02, B03, B04, NIR, SWIR1, SWIR2 |
| Opt.+F | 9 | + clear-sky shortwave radiation, PET, vapour-pressure deficit |
| Opt.+F+E | 14 | + elevation, topographic wetness index, available water capacity and its mask, aridity index |

## Released models

| Model | Stage | Input | Zenodo file |
|---|---|---|---|
| Prithvi-OpenET (Opt.) | 1, encoder frozen | Opt. | `prithvi_openet_opt.zip` |
| Prithvi-OpenET (Opt.+F) | 1 | Opt.+F | `prithvi_openet_opt_f.zip` |
| Prithvi-OpenET (Opt.+F+E) | 1 | Opt.+F+E | `prithvi_openet_opt_f_e.zip` |
| Prithvi-OpenET-Tower (Opt.) | 2, encoder fine-tuned | Opt. | `prithvi_openet_tower_opt.zip` |
| Prithvi-OpenET-Tower (Opt.+F) | 2 | Opt.+F | `prithvi_openet_tower_opt_f.zip` |
| Prithvi-OpenET-Tower (Opt.+F+E) | 2 | Opt.+F+E | `prithvi_openet_tower_opt_f_e.zip` |

Each model zip unpacks to a folder with `model.safetensors`, `config.json` (input channels, normalization statistics,
training summary), `README.md` (model card) and `SHA256SUMS`. The Prithvi-OpenET files do not contain the encoder, which
is unchanged from Prithvi-EO-2.0-300M-TL: on first use, `prithvi_openet` downloads it (about 1.3 GB) from Hugging Face
at the pinned revision `63adbd39c271da4c42f447e69b1a7c91a338cdc9` and checks its SHA-256. The Prithvi-OpenET-Tower files
contain their fine-tuned encoder.

The released models were trained after the paper's experiments with the same
settings on all available tiles and tower clusters, so they have no independent test set. **All accuracy values in the
paper come from the nine-fold cross-validated models.**

Held-out non-CONUS tower clusters, nine-fold cross-validation (paper Table 4), RMSE / bias in mm month⁻¹:

| Input | Prithvi-OpenET | Prithvi-OpenET-Tower |
|---|---|---|
| Opt. | 27.27 / +9.53 | 21.78 / +3.26 |
| Opt.+F | 25.33 / +8.47 | 20.15 / +3.26 |
| Opt.+F+E | 25.18 / +7.68 | 20.30 / +2.53 |

## Repository layout

```
prithvi_openet/          inference code: network, Prithvi-EO-2.0 encoder loading, input normalisation, prediction
Prithvi-OpenET/          README, examples/example_inference.ipynb
Prithvi-OpenET-Tower/    README, examples/example_inference.ipynb
examples/                three example chips used by both notebooks (two CONUS scenes, the FR-Lam tower chip)
scripts/                 run_example.py
PROVENANCE.md            how this code relates to the code that produced the paper
```

## Installation

Python 3.10, in three steps (see the header of `requirements.txt`):

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu   # or the build for your GPU
pip install -r requirements.txt
pip install --no-deps "terratorch @ git+https://github.com/IBM/terratorch.git@ca289f0663a84952247213958cf3727a07359a87"
```

terratorch 0.99.5 is installed without its dependency pins because it asks for torch <= 2.5, while the models were
trained and run with torch 2.7.1; the packages it needs are listed in `requirements.txt`.

## Quick start (CPU)

1. Download the model zips from Zenodo (to be released) and unzip them into `weights/` in the repository root (for example
   `unzip prithvi_openet_opt_f_e.zip -d weights/`). The Prithvi-OpenET models fetch their encoder from Hugging Face on
   first use.
2. Run a model on an example chip and check it against the stored prediction:

```bash
python scripts/run_example.py --weights weights/prithvi_openet_opt_f_e \
    --example examples/conus_T13TGE_2020-07.npz --out example_output --selftest
python scripts/run_example.py --weights weights/prithvi_openet_tower_opt_f_e \
    --example examples/france_lamasquere_2021-07.npz --out example_output --selftest
```

Or open the notebooks `Prithvi-OpenET/examples/example_inference.ipynb` and
`Prithvi-OpenET-Tower/examples/example_inference.ipynb`. From Python (repository root on `sys.path`):

```python
import numpy as np
from prithvi_openet import example_inputs, load_model, predict

model, config = load_model("weights/prithvi_openet_opt_f_e")
out = predict(model, example_inputs(np.load("examples/conus_T13TGE_2020-07.npz"), config))
out["et_mean"]        # 224 x 224 ET in mm month-1 (five-head mean); out["et_members"] holds the five heads
```

For other chips, `prithvi_openet.normalize` builds the model input from channel values in the stored units listed in
each `config.json` (`raster_channels`).

## Training

This repository is for running the released models. How they were trained is summarised in each model folder's README
and described in the paper.

## Data and licenses

Code: Apache License 2.0 (`LICENSE`). Example chips: CC BY-NC 4.0 plus the source terms in `examples/DATA_NOTICE.md`
(the topographic wetness index derives from MERIT Hydro, CC BY-NC 4.0). Model weights (Zenodo, to be released): CC BY-NC 4.0.
Documentation: CC BY 4.0. Third-party terms and the acknowledgments of the training data are in `NOTICE`.

## Citation

```bibtex
@article{choi_prithvi_openet,
  title   = {{Prithvi-OpenET}: Geospatial Foundation Model-Based 30 m Monthly Evapotranspiration Estimation beyond the Conterminous {United States} with {Landsat} and {Sentinel-2} Imagery},
  author  = {Choi, Euiyoung and Kim, Seulgi and Kim, Hyunglok},
  journal = {Remote Sensing of Environment},
  year    = {TODO},
  doi     = {TODO}
}
```
