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

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import solt as sl
import solt.transforms as slt


IMAGE_SIZE = 266
NUM_CLASSES = 6  # background + 5 organ classes


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable solt data augmentation")
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


def build_solt_transforms():
    return sl.Stream([
        slt.Rotate((-15, 15), padding="r"),
        slt.Shear(range_x=(-0.1, 0.1), range_y=(-0.1, 0.1), padding="r"),
        slt.Contrast(contrast_range=(0.6, 1.3)),
        slt.Brightness(brightness_range=(0.8, 1.2)),
        slt.Blur(p=0.1, blur_type="m", k_size=(3,)),
        slt.SaltAndPepper(p=0.1, gain_range=0.05),
        sl.SelectiveStream([
            sl.SelectiveStream([
                slt.CutOut(cutout_size=32),
                slt.CutOut(cutout_size=32),
                slt.CutOut(cutout_size=16),
                slt.CutOut(cutout_size=16),
                slt.CutOut(cutout_size=12),
                slt.CutOut(cutout_size=12),
                slt.CutOut(cutout_size=8),
                slt.CutOut(cutout_size=8),
            ], n=2),
            sl.Stream(),
        ], probs=[0.8, 0.2]),
    ])


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        pooled = self.pool(down)
        return down, pooled


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.bottle_neck = DoubleConv(512, 1024)
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        bottleneck = self.bottle_neck(p4)
        up_1 = self.up_convolution_1(bottleneck, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
        return self.out(up_4)


def pixel_decoder(encoded_pixels):
    encoded_pixels = str(encoded_pixels).split()
    starts = [int(encoded_pixels[i]) for i in range(0, len(encoded_pixels), 2)]
    lengths = [int(encoded_pixels[i]) for i in range(1, len(encoded_pixels), 2)]

    mask = np.zeros(IMAGE_SIZE * IMAGE_SIZE, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        start_idx = start - 1
        end_idx = start_idx + length
        mask[start_idx:end_idx] = 1

    return mask


image_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.ToDtype(torch.float32),
])


def apply_solt(solt_pipeline, img, mask):
    """Apply paired image/mask augmentations with solt."""
    if img.shape[:2] != mask.shape[:2]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    img_c = np.expand_dims(img, axis=-1) if img.ndim == 2 else img
    mask_c = np.expand_dims(mask, axis=-1) if mask.ndim == 2 else mask

    data = sl.core.DataContainer((img_c, mask_c), "IM")
    res = solt_pipeline(data, return_torch=False)

    aug_img = res.data[0].squeeze()
    aug_mask = res.data[1].squeeze()

    aug_mask = np.rint(aug_mask).astype(np.int64)
    aug_mask = np.clip(aug_mask, 0, NUM_CLASSES - 1)

    return aug_img, aug_mask


class CustomDataset(Dataset):
    def __init__(self, slice_contour_pairs, transform=None, mask_data_cache=None, solt_transform=None):
        self.slice_contour_pairs = slice_contour_pairs
        self.transform = transform
        self.mask_data_cache = mask_data_cache
        self.solt_transform = solt_transform

    def __len__(self):
        return len(self.slice_contour_pairs)

    def __getitem__(self, idx):
        slice_id, slice_path, contour_csv_path = self.slice_contour_pairs[idx]
        img = cv2.imread(str(slice_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {slice_path}")

        img = img.astype(np.float32)

        # min-max normalization to get values in 0-1 range
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        df = self.mask_data_cache[contour_csv_path]
        slice_rows = df[df["SliceID"] == slice_id]

        label_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)

        for _, row in slice_rows.iterrows():
            encoded_pixels = row["EncodedPixels"]
            if str(encoded_pixels) != "-1":
                mask = pixel_decoder(encoded_pixels)
                mask_2d = mask.reshape(IMAGE_SIZE, IMAGE_SIZE)
                organ_id = int(row["MaskTypeID"]) + 1
                label_map[mask_2d == 1] = organ_id

        if self.solt_transform is not None:
            img, label_map = apply_solt(self.solt_transform, img, label_map)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        target = torch.tensor(label_map, dtype=torch.long)
        return img, target


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


def collect_slice_pairs(dataset_path):
    slice_contour_pairs = []
    dataset_cases = sorted(os.listdir(dataset_path))

    for case_name in dataset_cases:
        case_path = Path(dataset_path) / case_name
        if not case_path.is_dir():
            continue

        case_days = sorted(os.listdir(case_path))
        for case_day in case_days:
            scans_path = case_path / case_day / "scans"
            contours_path = case_path / case_day / "contours"
            contour_csv_path = contours_path / "masks_rle.csv"

            if not scans_path.is_dir() or not contour_csv_path.is_file():
                continue

            for scan in sorted(os.listdir(scans_path)):
                slice_parts = scan.split("_")
                slice_id = slice_parts[0] + "_" + slice_parts[1]
                slice_path = scans_path / scan
                slice_contour_pairs.append((slice_id, slice_path, contour_csv_path))

    if not slice_contour_pairs:
        raise ValueError(f"No scan/mask pairs found inside: {dataset_path}")

    return slice_contour_pairs


def split_pairs_by_scan(slice_contour_pairs, seed):
    unique_scan_ids = list(set(pair[1].parent.parent.name for pair in slice_contour_pairs))
    random.seed(seed)
    random.shuffle(unique_scan_ids)

    num_scans = len(unique_scan_ids)
    train_idx = int(0.7 * num_scans)
    val_idx = train_idx + int(0.1 * num_scans)

    train_scan_ids = set(unique_scan_ids[:train_idx])
    val_scan_ids = set(unique_scan_ids[train_idx:val_idx])
    test_scan_ids = set(unique_scan_ids[val_idx:])

    train_pairs, val_pairs, test_pairs = [], [], []
    for pair in slice_contour_pairs:
        scan_id = pair[1].parent.parent.name
        if scan_id in train_scan_ids:
            train_pairs.append(pair)
        elif scan_id in val_scan_ids:
            val_pairs.append(pair)
        elif scan_id in test_scan_ids:
            test_pairs.append(pair)

    return train_pairs, val_pairs, test_pairs


def build_mask_cache(slice_contour_pairs):
    unique_contour_csv_paths = sorted({pair[2] for pair in slice_contour_pairs})
    return {csv_path: pd.read_csv(csv_path) for csv_path in tqdm(unique_contour_csv_paths, desc="Loading mask CSVs")}


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

    solt_transform = None if args.no_augment else build_solt_transforms()

    train_dataset = CustomDataset(
        train_pairs,
        image_transform,
        mask_data_cache=mask_dfs_cache,
        solt_transform=solt_transform,
    )
    val_dataset = CustomDataset(val_pairs, image_transform, mask_data_cache=mask_dfs_cache)
    test_dataset = CustomDataset(test_pairs, image_transform, mask_data_cache=mask_dfs_cache)

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
