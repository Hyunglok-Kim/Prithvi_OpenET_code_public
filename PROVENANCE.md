# Provenance

The released models were trained with an internal research codebase, commit b8e6d7b (2026-10-06).

This repository keeps only what is needed to run the released models:

- `prithvi_openet/`: the inference path of that codebase, rewritten as a small package: the network (encoder, head,
  refinement U-Net with five heads), loading of the Prithvi-EO-2.0-300M-TL encoder and the input normalisation. Training
  code, data loaders and training configuration are not included. Layer names are unchanged, so the released
  `model.safetensors` files load as they were saved. The Prithvi-OpenET weights do not contain the encoder, which is
  loaded unchanged from a pinned Hugging Face revision.
- `scripts/run_example.py`: runs a model on an example chip and compares the result with the prediction stored in the
  chip, which was made with the training codebase. All six models reproduce it exactly (`CHECKS.md`).
- `examples/`: three example chips with the stored predictions.

Checks before release are recorded in `CHECKS.md`.
