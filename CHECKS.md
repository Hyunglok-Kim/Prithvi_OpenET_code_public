# Release checks

Run on 2026-10-13 in a new Python 3.10.20 virtual environment on Linux (CPU only, empty Hugging Face cache), following
the installation steps of `README.md`, with the model zips prepared for the Zenodo record unzipped into `weights/`.

Installed: huggingface-hub==0.26.2 numpy==1.26.4 safetensors==0.4.5 timm==0.9.7 torch==2.7.1+cpu torchvision==0.22.1+cpu and terratorch at commit ca289f0663a84952247213958cf3727a07359a87 (`--no-deps`).

| Check | Result |
|---|---|
| `scripts/run_example.py --selftest`, 6 models × 3 example chips | 18/18 pass, maximum difference from the stored prediction 0.00 mm month⁻¹ |
| Prithvi-OpenET encoder download | Prithvi_EO_V2_300M_TL.pt fetched from revision 63adbd39c271da4c42f447e69b1a7c91a338cdc9, SHA-256 verified |
| `Prithvi-OpenET/examples/example_inference.ipynb` | runs without errors; three models match the stored predictions exactly |
| `Prithvi-OpenET-Tower/examples/example_inference.ipynb` | runs without errors; three models match the stored predictions exactly |
| Text search of the repository and of every file prepared for the Zenodo record | no internal paths or tool names; "B200" appears only as the GPU model of the training hardware |
