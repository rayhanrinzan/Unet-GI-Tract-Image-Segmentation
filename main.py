#!/usr/bin/env python3
"""Train a U-Net for GI tract MRI segmentation on a Slurm cluster.

This version is converted from a Colab notebook into a normal Python script:
- no google.colab imports
- no !pip install / !wandb login notebook commands
- dataset path comes from --dataset-root or GI_TRACT_DATASET_PATH
- model outputs are saved to --output-dir
- Weights & Biases logging is optional with --use-wandb
"""

#for visualizing scans/masks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from dataloader import (
    NUM_CLASSES,
    CustomDataset,
    build_mask_cache,
    collect_slice_pairs,
    train_transform,
    eval_transform,
    split_pairs_by_scan,
)
from model import UNet

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on GI tract MRI segmentation data")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.environ.get("GI_TRACT_DATASET_PATH"),
        help="Path to the dataset root. Can point either to the folder containing 'dataset/' or to 'dataset/' itself. "
             "If omitted, GI_TRACT_DATASET_PATH is used.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Folder for saved models/logs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="MRI Scans 5")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_dataset_path(root_path):
    """Resolve dataset path assuming folder name 'dataset' when root_path is its parent."""
    if not root_path:
        raise ValueError(
            "No dataset path provided. Set GI_TRACT_DATASET_PATH or pass --dataset-root. "
            "The path should point to the downloaded Kaggle folder or directly to its dataset/ folder."
        )

    root = Path(root_path).expanduser().resolve()
    dataset_path = root if root.name == "dataset" else root / "dataset"

    if not dataset_path.is_dir():
        raise ValueError(
            f"Dataset directory not found at: {dataset_path}\n"
            "Expected either --dataset-root /path/to/.../dataset or a folder containing dataset/."
        )

    return dataset_path


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
        targets_one_hot = targets_one_hot.reshape(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        intersection = (inputs * targets_one_hot).sum(2)
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum(2) + targets_one_hot.sum(2) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dice = DiceLoss()
        weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs, targets):
        target_long = targets.long()
        return self.ce(inputs, target_long) + self.dice(inputs, target_long)


def maybe_log(wandb_run, metrics):
    if wandb_run is not None:
        wandb_run.log(metrics)


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, wandb_run=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0.0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(x)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
            maybe_log(wandb_run, {"Train/Step_Loss": loss_val})

    return train_loss / max(num_batches, 1)


def evaluate(dataloader, model, loss_fn, device, split_name="Validation", wandb_run=None):
    num_batches = len(dataloader)
    model.eval()
    total_loss = 0.0
    total_correct_pixels = 0
    total_pixels = 0
    iou_sum = 0.0
    iou_count = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item()

            predicted_classes = pred.argmax(1)
            total_correct_pixels += (predicted_classes == y).sum().item()
            total_pixels += y.numel()

            for cls in range(1, NUM_CLASSES):
                inter = ((predicted_classes == cls) & (y == cls)).sum().item()
                union = ((predicted_classes == cls) | (y == cls)).sum().item()
                if union > 0:
                    iou_sum += inter / union
                    iou_count += 1

    avg_loss = total_loss / max(num_batches, 1)
    pixel_acc = total_correct_pixels / max(total_pixels, 1)
    avg_iou = iou_sum / max(iou_count, 1)

    print(f"{split_name} results:")
    print(f"  Avg loss: {avg_loss:>8f}")
    print(f"  Pixel Accuracy: {(100 * pixel_acc):>0.2f}%")
    print(f"  Mean IoU (Organs): {avg_iou:>8f}\n", flush=True)

    maybe_log(
        wandb_run,
        {
            f"{split_name}/Epoch_Loss": avg_loss,
            f"{split_name}/Pixel_Accuracy": pixel_acc,
            f"{split_name}/Mean_IoU": avg_iou,
        },
    )

    return avg_loss


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(args.dataset_root)
    print(f"Using dataset path: {dataset_path}")
    print(f"Saving outputs to: {output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    slice_contour_pairs = collect_slice_pairs(dataset_path)
    mask_dfs_cache = build_mask_cache(slice_contour_pairs)
    train_pairs, val_pairs, test_pairs = split_pairs_by_scan(slice_contour_pairs, args.seed)

    print(f"Total train samples: {len(train_pairs)}")
    print(f"Total validation samples: {len(val_pairs)}")
    print(f"Total test samples: {len(test_pairs)}")

    train_dataset = CustomDataset(
        train_pairs,
        eval_transform if args.no_augment else train_transform,
        mask_data_cache=mask_dfs_cache,
    )
    val_dataset = CustomDataset(val_pairs, eval_transform, mask_data_cache=mask_dfs_cache)
    test_dataset = CustomDataset(test_pairs, eval_transform, mask_data_cache=mask_dfs_cache)

    # TEMP DEBUG BLOCK: save a few transformed samples, then exit
    debug_dir = output_dir / "debug_samples"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(4):
        img, mask = train_dataset[i]
    
        img_np = img.squeeze().cpu().numpy()
        mask_np = mask.cpu().numpy()
    
        print(f"\nSample {i}")
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image min/max: {img_np.min():.4f}, {img_np.max():.4f}")
        print(f"Mask unique classes: {np.unique(mask_np)}")
    
        masked_organ = np.ma.masked_where(mask_np == 0, mask_np)
    
        plt.figure(figsize=(12, 4))
    
        plt.subplot(1, 3, 1)
        plt.imshow(img_np, cmap="gray")
        plt.title("Transformed Image")
        plt.axis("off")
    
        plt.subplot(1, 3, 2)
        plt.imshow(mask_np, cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1)
        plt.title("Transformed Mask")
        plt.axis("off")
    
        plt.subplot(1, 3, 3)
        plt.imshow(img_np, cmap="gray")
        plt.imshow(masked_organ, cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1, alpha=0.45)
        plt.title("Overlay")
        plt.axis("off")
    
        plt.tight_layout()
    
        save_path = debug_dir / f"sample_{i}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
    
        print(f"Saved: {save_path}")
    
    print("Finished saving transformed samples. Exiting before training.")
    return
    # end of temporary debugging logic
        

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device == "cuda",
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = CombinedLoss(device=device)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "learning_rate": args.learning_rate,
                "architecture": "UNet",
                "dataset": "GI Tract Image Segmentation",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "seed": args.seed,
                "augmentation": not args.no_augment,
            },
        )

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_save_path = output_dir / "unet_best_model.pth"
    final_save_path = output_dir / "unet_final_model.pth"

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_loss = train_one_epoch(train_dataloader, model, criterion, optimizer, device, wandb_run)
        train_loss_history.append(train_loss)
        maybe_log(wandb_run, {"Train/Epoch_Loss": train_loss, "epoch": epoch + 1})

        if len(val_dataloader) > 0:
            val_loss = evaluate(val_dataloader, model, criterion, device, split_name="Validation", wandb_run=wandb_run)
            val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_save_path)
                print(f"--> Validation loss improved to {best_val_loss:.6f}! Saved best model to {best_save_path}")
        else:
            print("Skipping validation: val_dataloader is empty.")

    print("Training Done!\n")
    print("-------------------------------\nFinal Evaluation on Test Set:")
    if len(test_dataloader) > 0:
        evaluate(test_dataloader, model, criterion, device, split_name="Test", wandb_run=wandb_run)
    else:
        print("Skipping test: test_dataloader is empty.")

    torch.save(model.state_dict(), final_save_path)
    print(f"Final model weights saved to {final_save_path}")

    if wandb_run is not None:
        wandb_run.save(str(final_save_path))
        if best_save_path.exists():
            wandb_run.save(str(best_save_path))
        wandb_run.finish()


if __name__ == "__main__":
    main()
