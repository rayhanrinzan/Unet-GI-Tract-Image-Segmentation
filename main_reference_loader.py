#!/usr/bin/env python3
"""Reference baseline reproduction for GI tract MRI segmentation on Slurm.

This script keeps the command-line style from my original main.py, but follows
the reference notebook's baseline much more closely:

- segmentation_models_pytorch U-Net
- EfficientNet encoder with ImageNet weights by default
- RGB image loading from grayscale MRI slices
- [3, H, W] multilabel masks for large_bowel, small_bowel, stomach
- sigmoid-style predictions, not argmax
- SoftBCEWithLogitsLoss + TverskyLoss
- StratifiedGroupKFold split by empty/non-empty masks and patient case
"""

import argparse
import copy
import gc
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
import segmentation_models_pytorch as smp


CLASS_NAMES = ["large_bowel", "small_bowel", "stomach"]
NUM_CLASSES = 3

# Dataset MaskTypeID mapping:
# 1 = large_bowel
# 3 = small_bowel
# 4 = stomach
CHANNEL_MAP = {
    1: 0,
    3: 1,
    4: 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reference-style U-Net baseline for GI tract MRI segmentation"
    )

    # Same core arguments as my original main.py
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.environ.get("GI_TRACT_DATASET_PATH"),
        help=(
            "Path to the dataset root. Can point either to the folder containing "
            "'dataset/' or to 'dataset/' itself. If omitted, GI_TRACT_DATASET_PATH is used."
        ),
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

    # Reference-baseline options
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b1")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="CosineAnnealingLR",
        choices=["CosineAnnealingLR", "CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "ExponentialLR", "None"],
    )
    parser.add_argument("--debug", action="store_true", help="Use a small non-empty subset for quick debugging")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional limit for training samples")
    parser.add_argument("--valid-limit", type=int, default=None, help="Optional limit for validation samples")
    parser.add_argument(
        "--mask-cache-dir",
        type=str,
        default=None,
        help="Folder for generated .npy masks. Defaults to output_dir/reference_masks.",
    )

    return parser.parse_args()


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("> SEEDING DONE")


def resolve_dataset_path(root_path):
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


def rle_decode(mask_rle, shape):
    """Decode run-length encoding into a binary mask."""
    s = str(mask_rle).split()

    if len(s) == 0:
        return np.zeros(shape, dtype=np.uint8)

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)


def load_img(path):
    """Load grayscale MRI slice as 3-channel RGB-style image."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = np.tile(img[..., None], [1, 1, 3])
    img = img.astype("float32")

    mx = np.max(img)
    if mx:
        img /= mx

    return img


def load_msk(path):
    """Load precomputed 3-channel .npy mask."""
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk


def collect_reference_rows(dataset_path, mask_cache_dir):
    """Build a reference-style dataframe and generate .npy masks from the RLE CSVs."""
    rows = []
    mask_cache_dir.mkdir(parents=True, exist_ok=True)

    case_paths = sorted(Path(dataset_path).iterdir())

    for case_path in tqdm(case_paths, desc="Scanning cases"):
        if not case_path.is_dir():
            continue

        case_name = case_path.name
        case_number = "".join(ch for ch in case_name if ch.isdigit())

        for case_day_path in sorted(case_path.iterdir()):
            if not case_day_path.is_dir():
                continue

            scans_path = case_day_path / "scans"
            contours_path = case_day_path / "contours"

            contour_csv_path = contours_path / "masks_rle.csv"
            if not contour_csv_path.is_file():
                contour_csv_path = contours_path / "mask_rle.csv"

            if not scans_path.is_dir() or not contour_csv_path.is_file():
                continue

            mask_df = pd.read_csv(contour_csv_path)

            for scan_path in sorted(scans_path.iterdir()):
                if not scan_path.is_file():
                    continue

                slice_parts = scan_path.name.split("_")
                if len(slice_parts) < 2:
                    continue

                slice_id = slice_parts[0] + "_" + slice_parts[1]

                img = cv2.imread(str(scan_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Skipping unreadable image: {scan_path}")
                    continue

                height, width = img.shape[:2]
                slice_rows = mask_df[mask_df["SliceID"] == slice_id]

                mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

                for _, row in slice_rows.iterrows():
                    mask_type_id = int(row["MaskTypeID"])
                    encoded_pixels = row["EncodedPixels"]

                    if mask_type_id not in CHANNEL_MAP:
                        continue

                    if pd.isna(encoded_pixels) or str(encoded_pixels) == "-1":
                        continue

                    channel = CHANNEL_MAP[mask_type_id]
                    mask[..., channel] = rle_decode(str(encoded_pixels), (height, width)) * 255

                relative_key = f"{case_name}_{case_day_path.name}_{scan_path.stem}"
                mask_path = mask_cache_dir / f"{relative_key}.npy"
                np.save(mask_path, mask)

                rows.append(
                    {
                        "id": f"{case_day_path.name}_{slice_id}",
                        "case": int(case_number) if case_number else -1,
                        "case_name": case_name,
                        "day_name": case_day_path.name,
                        "slice_id": slice_id,
                        "image_path": str(scan_path),
                        "mask_path": str(mask_path),
                        "height": height,
                        "width": width,
                        "empty": bool(mask.sum() == 0),
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No usable image/mask rows found inside: {dataset_path}")

    return df


class BuildDataset(Dataset):
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


def get_transforms(img_size, no_augment=False):
    valid_tfms = A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
        ],
        p=1.0,
    )

    if no_augment:
        return valid_tfms, valid_tfms

    train_tfms = A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.05,
                rotate_limit=10,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                ],
                p=0.25,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=img_size // 20,
                max_width=img_size // 20,
                min_holes=5,
                fill_value=0,
                mask_fill_value=0,
                p=0.5,
            ),
        ],
        p=1.0,
    )

    return train_tfms, valid_tfms


def create_folds(df, n_folds, seed):
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    df = df.copy()
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df["empty"], groups=df["case_name"])):
        df.loc[val_idx, "fold"] = fold

    return df


def prepare_loaders(df, fold, args):
    train_df = df.query("fold != @fold").reset_index(drop=True)
    valid_df = df.query("fold == @fold").reset_index(drop=True)

    if args.debug:
        train_df = train_df.query("empty == False").head(160).reset_index(drop=True)
        valid_df = valid_df.query("empty == False").head(96).reset_index(drop=True)

    if args.train_limit is not None:
        train_df = train_df.head(args.train_limit).reset_index(drop=True)

    if args.valid_limit is not None:
        valid_df = valid_df.head(args.valid_limit).reset_index(drop=True)

    train_tfms, valid_tfms = get_transforms(args.img_size, no_augment=args.no_augment)

    train_dataset = BuildDataset(train_df, transforms=train_tfms)
    valid_dataset = BuildDataset(valid_df, transforms=valid_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    return train_loader, valid_loader, train_df, valid_df


def build_model(args, device):
    model = smp.Unet(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    model.to(device)
    return model


JaccardLoss = smp.losses.JaccardLoss(mode="multilabel")
DiceLoss = smp.losses.DiceLoss(mode="multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode="multilabel", per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)


def criterion(y_pred, y_true):
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)

    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)

    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)

    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)

    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def fetch_scheduler(optimizer, args, train_loader_len):
    scheduler_name = args.scheduler

    if scheduler_name == "None":
        return None

    if scheduler_name == "CosineAnnealingLR":
        t_max = int(train_loader_len * args.epochs) + 50
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=args.min_lr,
        )

    if scheduler_name == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=25,
            eta_min=args.min_lr,
        )

    if scheduler_name == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=7,
            threshold=0.0001,
            min_lr=args.min_lr,
        )

    if scheduler_name == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    return None


def maybe_log(wandb_run, metrics):
    if wandb_run is not None:
        wandb_run.log(metrics)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, wandb_run=None):
    model.train()

    scaler = amp.GradScaler(enabled=(device == "cuda"))

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train ")

    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=(device == "cuda")):
            y_pred = model(images)
            loss = criterion(y_pred, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None and not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]

        probs = torch.sigmoid(y_pred.detach())
        batch_iou = iou_coef(masks.detach(), probs).detach().cpu().item()

        pbar.set_postfix(
            train_loss=f"{epoch_loss:0.4f}",
            train_iou=f"{batch_iou:0.4f}",
            lr=f"{current_lr:0.5f}",
            gpu_mem=f"{mem:0.2f} GB",
        )

    torch.cuda.empty_cache()
    gc.collect()

    maybe_log(wandb_run, {"Train Loss": epoch_loss, "epoch": epoch})

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch, wandb_run=None):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid ")

    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()

        val_scores.append([val_dice, val_jaccard])

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            valid_loss=f"{epoch_loss:0.4f}",
            valid_dice=f"{float(val_dice):0.4f}",
            valid_iou=f"{float(val_jaccard):0.4f}",
            gpu_memory=f"{mem:0.2f} GB",
        )

    val_scores = np.mean(val_scores, axis=0)

    torch.cuda.empty_cache()
    gc.collect()

    val_dice, val_jaccard = val_scores

    maybe_log(
        wandb_run,
        {
            "Valid Loss": epoch_loss,
            "Valid Dice": float(val_dice),
            "Valid Jaccard": float(val_jaccard),
            "epoch": epoch,
        },
    )

    return epoch_loss, val_scores


def save_debug_overlay(dataset, output_dir):
    if len(dataset) == 0:
        return

    import matplotlib.pyplot as plt

    img, mask = dataset[0]

    img_np = img.permute(1, 2, 0).numpy()
    mask_np = mask.sum(dim=0).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.imshow(mask_np, alpha=0.4)
    plt.title("Reference Baseline: Image + Mask Overlay")
    plt.axis("off")

    out_path = output_dir / "debug_reference_baseline_overlay.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved debug overlay to: {out_path}")


def run_training(model, optimizer, scheduler, train_loader, valid_loader, device, args, output_dir, wandb_run=None):
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_jaccard = -np.inf
    best_epoch = -1

    history = defaultdict(list)

    best_path = output_dir / f"reference_baseline_best_fold{args.fold}.pth"
    last_path = output_dir / f"reference_baseline_last_fold{args.fold}.pth"

    for epoch in range(1, args.epochs + 1):
        gc.collect()

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            wandb_run=wandb_run,
        )

        val_loss, val_scores = valid_one_epoch(
            model,
            valid_loader,
            device=device,
            epoch=epoch,
            wandb_run=wandb_run,
        )

        val_dice, val_jaccard = val_scores

        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        history["Valid Dice"].append(float(val_dice))
        history["Valid Jaccard"].append(float(val_jaccard))

        if scheduler is not None and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        maybe_log(
            wandb_run,
            {
                "LR": current_lr,
                "Best Dice So Far": float(best_dice),
                "Best Jaccard So Far": float(best_jaccard),
            },
        )

        print(f"Valid Dice: {float(val_dice):0.4f} | Valid Jaccard: {float(val_jaccard):0.4f}")

        if val_dice >= best_dice:
            print(f"Valid Dice improved ({best_dice:0.4f} -> {float(val_dice):0.4f})")
            best_dice = float(val_dice)
            best_jaccard = float(val_jaccard)
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to: {best_path}")

            if wandb_run is not None:
                wandb_run.summary["Best Dice"] = best_dice
                wandb_run.summary["Best Jaccard"] = best_jaccard
                wandb_run.summary["Best Epoch"] = best_epoch
                wandb_run.save(str(best_path))

        torch.save(model.state_dict(), last_path)

    end = time.time()
    time_elapsed = end - start

    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Best Jaccard: {best_jaccard:.4f}")
    print(f"Best Epoch: {best_epoch}")

    model.load_state_dict(best_model_wts)

    return model, history


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(args.dataset_root)

    if args.mask_cache_dir is None:
        mask_cache_dir = output_dir / "reference_baseline_masks"
    else:
        mask_cache_dir = Path(args.mask_cache_dir).expanduser().resolve()

    print(f"Using dataset path: {dataset_path}")
    print(f"Saving outputs to: {output_dir}")
    print(f"Saving generated masks to: {mask_cache_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = collect_reference_rows(dataset_path, mask_cache_dir)

    print("\nDataset summary:")
    print(df[["id", "case_name", "image_path", "mask_path", "empty"]].head())
    print(df["empty"].value_counts())

    df = create_folds(df, args.n_folds, args.seed)

    print("\nFold distribution:")
    print(df.groupby(["fold", "empty"])["id"].count())

    train_loader, valid_loader, train_df, valid_df = prepare_loaders(df, args.fold, args)

    print(f"\nFold: {args.fold}")
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")

    print("\n--- DATASET SANITY CHECK ---")
    sample_dataset = train_loader.dataset

    for i in range(min(10, len(sample_dataset))):
        img, mask = sample_dataset[i]
        print(f"sample {i}")
        print("img shape:", img.shape)
        print("mask shape:", mask.shape)
        print("large_bowel pixels:", mask[0].sum().item())
        print("small_bowel pixels:", mask[1].sum().item())
        print("stomach pixels:", mask[2].sum().item())
        print()

    save_debug_overlay(sample_dataset, output_dir)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name
            or f"reference-baseline-fold-{args.fold}-{args.encoder_name}-{args.img_size}",
            config=vars(args),
        )

    model = build_model(args, device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = fetch_scheduler(
        optimizer,
        args=args,
        train_loader_len=len(train_loader),
    )

    model, history = run_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        args=args,
        output_dir=output_dir,
        wandb_run=wandb_run,
    )

    history_df = pd.DataFrame(history)
    history_path = output_dir / f"reference_baseline_history_fold{args.fold}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved history to: {history_path}")

    if wandb_run is not None:
        wandb_run.save(str(history_path))
        wandb_run.finish()


if __name__ == "__main__":
    main()
