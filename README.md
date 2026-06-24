# Unet-GI-Tract-Image-Segmentation
This repository contains a script for GI Tract Image Segmentation. It utilizes a U-Net architecture, processes data from the UW-Madison GI Tract Image Segmentation dataset, and includes custom data loading, augmentation (using solt), and training routines with Weights &amp; Biases integration for experiment tracking.

## Dataset download (run once)
Run:

```bash
python download_dataset.py
```

This downloads the dataset and writes the local dataset path to `dataset_path.txt`.

Before training, set:

```bash
export GI_TRACT_DATASET_PATH="<downloaded_dataset_path>"
```

Or (bash/zsh), load it directly from the generated file:

```bash
export GI_TRACT_DATASET_PATH="$(cat dataset_path.txt)"
```
## Entry point
The main script is now `main.py`.

## Run
This project was originally written as a Colab-style notebook export, so you may need to remove notebook-only commands like `!pip install ...`, `from google.colab import drive`, and `!wandb login --relogin` before running it as a standalone Python script.
