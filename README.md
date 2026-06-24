# Unet-GI-Tract-Image-Segmentation
This repository contains a script for GI Tract Image Segmentation. It utilizes a U-Net architecture, processes data from the UW-Madison GI Tract Image Segmentation dataset, and includes custom data loading, augmentation (using solt), and training routines with Weights &amp; Biases integration for experiment tracking.

## Entry point
The main script is now `main.py`.

## Run
This project was originally written as a Colab-style notebook export, so you may need to remove notebook-only commands like `!pip install ...`, `from google.colab import drive`, and `!wandb login --relogin` before running it as a standalone Python script.
