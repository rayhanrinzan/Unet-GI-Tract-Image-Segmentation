#!/usr/bin/env python3
"""Experimental reference-style loader for GI tract MRI segmentation.

This file intentionally keeps your existing split/Slurm/main structure, but comments out
your original CustomDataset path and uses a reference-style image/mask/dataset pipeline:

- image_path + mask_path dataframe
- precomputed .npy masks
- RGB image loading like the reference notebook
- [3, H, W] binary masks for large_bowel, small_bowel, stomach
- BCEWithLogitsLoss + sigmoid predictions
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# EDIT 1:
# Old import commented out because this experiment does NOT use your CustomDataset,
# NUM_CLASSES, train_transform, or eval_transform.
#
# from dataloader import (
#     NUM_CLASSES,
#     CustomDataset,
#     build_mask_cache,
#     collect_slice_pairs,
#     train_transform,
#     eval_transform,
#     split_pairs_by_scan,
# )

# Keep only your metadata/split helpers.
from dataloader import (
    build_mask_cache,
    collect_slice_pairs,
    split_pairs_by_scan,
)

from model import UNet


REFERENCE_NUM_CLASSES = 3  # large_bowel, small_bowel, stomach


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
    parser.add_argument("--no-augment", action="store_true", help="Kept for CLI compatibility; this reference loader uses no transforms.")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="MRI Scans 5")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # For this experiment, skip full test by default so we do not generate thousands
    # of .npy masks every quick debug run.
    parser.add_argument("--run-test", action="store_true", help="Generate reference masks for the full test split and evaluate it.")

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


# EDIT 2:
# Reference notebook image/mask/RLE sections pasted here.
# This keeps the reference-style RGB image loading and .npy mask loading.


def load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = np.tile(img[..., None], [1, 1, 3])  # gray to RGB, same as reference notebook
    img = img.astype("float32")  # original is uint16

    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]

    return img


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk


def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted as start length
    shape: (height, width)
    returns: binary numpy array, 1 = mask, 0 = background
    """
    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)


# EDIT 3:
# Bridge from your current slice_contour_pairs + masks_rle.csv setup
# into the reference notebook's expected dataframe format:
# image_path + mask_path.
#
# This is the only custom bridge code needed because the reference notebook used
# a separate precomputed .npy mask dataset, while your repo currently has RLE CSVs.


def build_reference_mask(slice_id, contour_csv_path, height, width, mask_dfs_cache):
    df = mask_dfs_cache[contour_csv_path]
    rows = df[df["SliceID"] == slice_id]

    # [H, W, 3]
    # channel 0 = large_bowel
    # channel 1 = small_bowel
    # channel 2 = stomach
    mask = np.zeros((height, width, REFERENCE_NUM_CLASSES), dtype=np.uint8)

    channel_map = {
        1: 0,  # large_bowel
        3: 1,  # small_bowel
        4: 2,  # stomach
    }

    for _, row in rows.iterrows():
        mask_type_id = int(row["MaskTypeID"])
        encoded_pixels = row["EncodedPixels"]

        if mask_type_id not in channel_map:
            continue

        if pd.isna(encoded_pixels) or str(encoded_pixels) == "-1":
            continue

        channel = channel_map[mask_type_id]
        mask[..., channel] = rle_decode(str(encoded_pixels), (height, width)) * 255

    return mask


def make_reference_df(pairs, mask_dfs_cache, output_dir, split_name):
    mask_dir = output_dir / "reference_masks" / split_name
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, (slice_id, slice_path, contour_csv_path) in enumerate(pairs):
        img = cv2.imread(str(slice_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipping unreadable image: {slice_path}")
            continue

        height, width = img.shape[:2]

        mask = build_reference_mask(
            slice_id=slice_id,
            contour_csv_path=contour_csv_path,
            height=height,
            width=width,
            mask_dfs_cache=mask_dfs_cache,
        )

        # Include idx to avoid collisions across case/day folders with the same slice filename.
        mask_path = mask_dir / f"{idx:06d}_{slice_path.stem}.npy"
        np.save(mask_path, mask)

        rows.append({
            "id": f"{contour_csv_path.parent.parent.parent.name}_{contour_csv_path.parent.parent.name}_{slice_id}",
            "image_path": str(slice_path),
            "mask_path": str(mask_path),
            "height": height,
            "width": width,
            "empty": mask.sum() == 0,
        })

    return pd.DataFrame(rows)


# EDIT 4:
# Reference notebook dataset section pasted here.
# It expects a dataframe with image_path and mask_path columns.


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.msk_paths = df["mask_path"].tolist() if label else None
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)

            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]

            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))

            return torch.tensor(img), torch.tensor(msk)

        if self.transforms:
            data = self.transforms(image=img)
            img = data["image"]

        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img)


# EDIT 5:
# Old softmax/CrossEntropy/DiceLoss setup is intentionally not used in this file.
# The reference-style mask format is [B, 3, H, W], so the compatible loss is BCEWithLogitsLoss.
#
# Old DiceLoss and CombinedLoss are left out instead of copied/commented to keep
# this experimental file easy to run.


def maybe_log(wandb_run, metrics):
    if wandb_run is not None:
        wandb_run.log(metrics)


def multilabel_iou(pred_masks, y):
    """
    pred_masks and y: [B, 3, H, W]
    returns mean IoU across channels that have nonzero union.
    """
    intersection = (pred_masks * y).sum(dim=(0, 2, 3))
    union = ((pred_masks + y) > 0).float().sum(dim=(0, 2, 3))

    valid = union > 0
    if valid.any():
        return (intersection[valid] / union[valid]).mean().item()

    return 0.0


# EDIT 6:
# Training loop now uses sigmoid thresholding instead of argmax.


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, wandb_run=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    train_loss = 0.0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 1 == 0:
            loss_val = loss.item()
            current = batch * len(x)

            pred_masks = (torch.sigmoid(pred) > 0.5).float()
            batch_iou = multilabel_iou(pred_masks, y)

            print("pred channel sums:", pred_masks.sum(dim=(0, 2, 3)).detach().cpu())
            print("true channel sums:", y.sum(dim=(0, 2, 3)).detach().cpu())

            print(
                f"loss: {loss_val:>7f}  IoU: {batch_iou:>7f}  [{current:>5d}/{size:>5d}]",
                flush=True,
            )

            maybe_log(wandb_run, {
                "Train/Step_Loss": loss_val,
                "Train/Step_IoU": batch_iou,
            })

    return train_loss / max(num_batches, 1)


# EDIT 7:
# Evaluation also uses sigmoid thresholding and reports per-channel IoU.


def evaluate(dataloader, model, loss_fn, device, split_name="Validation", wandb_run=None):
    num_batches = len(dataloader)

    model.eval()

    total_loss = 0.0
    total_correct_pixels = 0
    total_pixels = 0

    iou_sum = torch.zeros(REFERENCE_NUM_CLASSES, device=device)
    iou_count = torch.zeros(REFERENCE_NUM_CLASSES, device=device)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            total_loss += loss_fn(pred, y.float()).item()

            pred_masks = (torch.sigmoid(pred) > 0.5).float()

            total_correct_pixels += (pred_masks == y).sum().item()
            total_pixels += y.numel()

            intersection = (pred_masks * y).sum(dim=(0, 2, 3))
            union = ((pred_masks + y) > 0).float().sum(dim=(0, 2, 3))

            valid = union > 0
            iou_sum[valid] += intersection[valid] / union[valid]
            iou_count[valid] += 1

    avg_loss = total_loss / max(num_batches, 1)
    pixel_acc = total_correct_pixels / max(total_pixels, 1)

    per_class_iou = iou_sum / torch.clamp(iou_count, min=1)
    avg_iou = per_class_iou.mean().item()

    print(f"{split_name} results:")
    print(f"  Avg loss: {avg_loss:>8f}")
    print(f"  Pixel Accuracy: {(100 * pixel_acc):>0.2f}%")
    print(f"  Large Bowel IoU: {per_class_iou[0].item():>8f}")
    print(f"  Small Bowel IoU: {per_class_iou[1].item():>8f}")
    print(f"  Stomach IoU: {per_class_iou[2].item():>8f}")
    print(f"  Mean IoU: {avg_iou:>8f}\n", flush=True)

    maybe_log(
        wandb_run,
        {
            f"{split_name}/Epoch_Loss": avg_loss,
            f"{split_name}/Pixel_Accuracy": pixel_acc,
            f"{split_name}/Large_Bowel_IoU": per_class_iou[0].item(),
            f"{split_name}/Small_Bowel_IoU": per_class_iou[1].item(),
            f"{split_name}/Stomach_IoU": per_class_iou[2].item(),
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

    def has_eval_mask(pair):
        slice_id, _, contour_csv_path = pair
        df = mask_dfs_cache[contour_csv_path]
        rows = df[df["SliceID"] == slice_id]

        # MaskTypeID: 1 large_bowel, 3 small_bowel, 4 stomach
        rows = rows[rows["MaskTypeID"].isin([1, 3, 4])]

        if rows.empty:
            return False

        encoded = rows["EncodedPixels"]
        return encoded.notna().any() and (encoded.astype(str) != "-1").any()

    train_pairs = [p for p in train_pairs if has_eval_mask(p)]

    # temporarily trying to overfit
    train_pairs = train_pairs[:64]
    val_pairs = train_pairs[:64]

    # EDIT 8:
    # Old CustomDataset creation commented out.
    #
    # train_dataset = CustomDataset(
    #     train_pairs,
    #     eval_transform if args.no_augment else train_transform,
    #     mask_data_cache=mask_dfs_cache,
    # )
    # val_dataset = CustomDataset(val_pairs, eval_transform, mask_data_cache=mask_dfs_cache)
    # test_dataset = CustomDataset(test_pairs, eval_transform, mask_data_cache=mask_dfs_cache)

    # New reference-style dataframe + BuildDataset creation.
    train_df = make_reference_df(train_pairs, mask_dfs_cache, output_dir, split_name="train")
    val_df = make_reference_df(val_pairs, mask_dfs_cache, output_dir, split_name="val")

    if args.run_test:
        test_df = make_reference_df(test_pairs, mask_dfs_cache, output_dir, split_name="test")
    else:
        print("Skipping full test mask generation for this debug run. Use --run-test to enable it.")
        test_df = pd.DataFrame(columns=["id", "image_path", "mask_path", "height", "width", "empty"])

    train_dataset = BuildDataset(train_df)
    val_dataset = BuildDataset(val_df)
    test_dataset = BuildDataset(test_df)

    print("\n--- DATASET SANITY CHECK ---")
    for i in range(min(10, len(train_dataset))):
        img, target = train_dataset[i]
        print(f"sample {i}")
        print("img shape:", img.shape)
        print("target shape:", target.shape)
        print("large_bowel pixels:", target[0].sum().item())
        print("small_bowel pixels:", target[1].sum().item())
        print("stomach pixels:", target[2].sum().item())
        print()

    print(f"Total train samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")

    # EDIT 9:
    # Reference-style overlay debug image.
    # Since masks are [3, H, W], sum across channels for an overlay.
    if len(train_dataset) > 0:
        import matplotlib.pyplot as plt

        img, target = train_dataset[0]

        plt.figure(figsize=(6, 6))
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.imshow(target.sum(dim=0).numpy(), alpha=0.4)
        plt.title("Reference Loader: Image + Mask Overlay")
        plt.axis("off")
        plt.savefig(output_dir / "debug_reference_overlay.png")
        plt.close()

        print(f"Saved debug overlay to: {output_dir / 'debug_reference_overlay.png'}")

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

    # EDIT 10:
    # Reference loader images are RGB [3, H, W], so in_channels=3.
    # Reference masks are 3-channel foreground masks, so num_classes=3.
    model = UNet(in_channels=3, num_classes=REFERENCE_NUM_CLASSES).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # EDIT 11:
    # Reference-style 3-channel binary target uses BCEWithLogitsLoss.
    criterion = nn.BCEWithLogitsLoss()

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
                "loader": "reference_style_npy_masks",
                "num_classes": REFERENCE_NUM_CLASSES,
                "input_channels": 3,
            },
        )

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_save_path = output_dir / "unet_reference_loader_best_model.pth"
    final_save_path = output_dir / "unet_reference_loader_final_model.pth"

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
        print("Skipping test: test_dataloader is empty. Use --run-test to enable full test generation/evaluation.")

    torch.save(model.state_dict(), final_save_path)
    print(f"Final model weights saved to {final_save_path}")

    if wandb_run is not None:
        wandb_run.save(str(final_save_path))
        if best_save_path.exists():
            wandb_run.save(str(best_save_path))
        wandb_run.finish()


if __name__ == "__main__":
    main()
