#!/usr/bin/env python3
"""Train a U-Net for GI tract MRI segmentation on a Slurm cluster.

This version is converted from a Colab notebook into a normal Python script:
- no google.colab imports
- no !pip install / !wandb login notebook commands
- dataset path comes from --dataset-root or GI_TRACT_DATASET_PATH
- model outputs are saved to --output-dir
- Weights & Biases logging is optional with --use-wandb
"""

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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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

class CombinedLoss(nn.Module):
    def __init__(self, device, ce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        # EXPERIMENT: weighted CE + foreground Dice for sparse organ masks.
        # To go back to the baseline, use nn.CrossEntropyLoss() and return only CE below.
        class_weights = torch.tensor([0.05, 1.0, 1.5, 1.0], dtype=torch.float32, device=device)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        targets = targets.long()
        ce_loss = self.ce(inputs, targets)

        probs = F.softmax(inputs, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Dice is computed only on organ classes, not background, so it tracks IoU better.
        probs_fg = probs[:, 1:]
        target_fg = target_one_hot[:, 1:]
        dims = (0, 2, 3)
        intersection = (probs_fg * target_fg).sum(dims)
        denominator = probs_fg.sum(dims) + target_fg.sum(dims)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def maybe_log(wandb_run, metrics):
    if wandb_run is not None:
        wandb_run.log(metrics)


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, wandb_run=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0.0

    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 1 == 0:
            loss_val = loss.item()
            current = batch * len(x)
        
            predicted_classes = pred.argmax(1)
            print("pred unique:", torch.unique(predicted_classes, return_counts=True))
            print("true unique:", torch.unique(y, return_counts=True))
        
            intersection = ((predicted_classes == y) & (y > 0)).sum().item()
            union = ((predicted_classes > 0) | (y > 0)).sum().item()
        
            batch_iou = intersection / union if union > 0 else 0.0
        
            print(
                f"loss: {loss_val:>7f}  IoU: {batch_iou:>7f}  [{current:>5d}/{size:>5d}]",
                flush=True,
            )
        
            maybe_log(wandb_run, {
                "Train/Step_Loss": loss_val,
                "Train/Step_IoU": batch_iou,
            })
            
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

            EVAL_CLASSES = [1, 2, 3]
            for cls in EVAL_CLASSES:    
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

    return avg_loss, avg_iou


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
   
    def has_eval_mask(pair):
        slice_id, _, contour_csv_path = pair
        df = mask_dfs_cache[contour_csv_path]
        rows = df[df["SliceID"] == slice_id]
    
        # MaskTypeID: 1 large_bowel, 3 small_bowel, 4 stomach
        rows = rows[rows["MaskTypeID"].isin([1, 3, 4])]
    
        return (rows["EncodedPixels"].astype(str) != "-1").any()
    
    train_pairs = [p for p in train_pairs if has_eval_mask(p)]
    # temporarily trying to overfit
    # train_pairs = train_pairs[:64]
    # val_pairs = train_pairs[:64]

    train_dataset = CustomDataset(
        train_pairs,
        eval_transform if args.no_augment else train_transform,
        mask_data_cache=mask_dfs_cache,
    )
    
    val_dataset = CustomDataset(val_pairs, eval_transform, mask_data_cache=mask_dfs_cache)
    test_dataset = CustomDataset(test_pairs, eval_transform, mask_data_cache=mask_dfs_cache)

    for i in range(10):
        img, target = train_dataset[i]
        print(f"sample {i}")
        print("img shape:", img.shape)
        print("target shape:", target.shape)
        print("target unique:", torch.unique(target, return_counts=True))
        print()
    
    print(f"Total train samples: {len(train_pairs)}")
    print(f"Total validation samples: {len(val_pairs)}")
    print(f"Total test samples: {len(test_pairs)}")
    import matplotlib.pyplot as plt

    img, target = train_dataset[0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img.squeeze(), cmap="gray")
    plt.imshow(target, alpha=0.4)
    plt.title("Image + Label Map Overlay")
    plt.axis("off")
    plt.savefig(output_dir / "debug_overlay.png")
    plt.close()

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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    criterion = CombinedLoss(device=device)
    print("running train.py")
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
    best_val_iou = -1.0
    best_save_path = output_dir / "unet_best_model.pth"
    final_save_path = output_dir / "unet_final_model.pth"

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_loss = train_one_epoch(train_dataloader, model, criterion, optimizer, device, wandb_run)
        train_loss_history.append(train_loss)
        maybe_log(wandb_run, {"Train/Epoch_Loss": train_loss, "epoch": epoch + 1})

        if len(val_dataloader) > 0:
            val_loss, val_iou = evaluate(val_dataloader, model, criterion, device, split_name="Validation", wandb_run=wandb_run)
            val_loss_history.append(val_loss)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), best_save_path)
                print(f"--> Validation IoU improved to {best_val_iou:.6f}! Saved best model to {best_save_path}")
        else:
            print("Skipping validation: val_dataloader is empty.")

    print("Training Done!\n")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model weights saved to {final_save_path}")

    print("-------------------------------\nFinal Evaluation on Test Set:")
    if len(test_dataloader) > 0:
        if best_save_path.exists():
            model.load_state_dict(torch.load(best_save_path, map_location=device))
            print(f"Loaded best validation-IoU model from {best_save_path} for test evaluation.")
        evaluate(test_dataloader, model, criterion, device, split_name="Test", wandb_run=wandb_run)
    else:
        print("Skipping test: test_dataloader is empty.")

    if wandb_run is not None:
        wandb_run.save(str(final_save_path))
        if best_save_path.exists():
            wandb_run.save(str(best_save_path))
        wandb_run.finish()

if __name__ == "__main__":
    main()
