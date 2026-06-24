# Unet-GI-Tract-Image-Segmentation
This repository contains a script for GI Tract Image Segmentation. It utilizes a U-Net architecture, processes data from the UW-Madison GI Tract Image Segmentation dataset, and includes custom data loading, augmentation (using solt), and training routines with Weights &amp; Biases integration for experiment tracking.

## Dataset download (run once)
Run:

```bash
python /home/runner/work/Unet-GI-Tract-Image-Segmentation/Unet-GI-Tract-Image-Segmentation/download_dataset.py
```

This downloads the dataset and writes the local dataset path to `dataset_path.txt`.

Before training, set:

```bash
export GI_TRACT_DATASET_PATH="<downloaded_dataset_path>"
```
