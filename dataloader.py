from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


IMAGE_SIZE = 266
NUM_CLASSES = 6  # background + 5 organ classes


def pixel_decoder(encoded_pixels, height, width):
    encoded_pixels = str(encoded_pixels).split()
    starts = [int(encoded_pixels[i]) for i in range(0, len(encoded_pixels), 2)]
    lengths = [int(encoded_pixels[i]) for i in range(1, len(encoded_pixels), 2)]

    mask = np.zeros(height * width, dtype=np.uint8)

    for start, length in zip(starts, lengths):
        start_idx = start - 1
        end_idx = start_idx + length
        mask[start_idx:end_idx] = 1

    return mask.reshape(height, width)


train_transform = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.RandomRotation(
        degrees=(-10, 10),
        interpolation=InterpolationMode.BILINEAR,
        fill={tv_tensors.Image: 0.0, tv_tensors.Mask: 0},
    ),
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomVerticalFlip(p=0.3),
])

eval_transform = v2.Compose([
    v2.Resize(IMAGE_SIZE, IMAGE_SIZE, antialias=True),
])

class CustomDataset(Dataset):
    def __init__(self, slice_contour_pairs, transform=None, mask_data_cache=None):
        self.slice_contour_pairs = slice_contour_pairs
        self.transform = transform
        self.mask_data_cache = mask_data_cache

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
    
        # IMPORTANT: use original image shape for RLE decoding
        height, width = img.shape[:2]
    
        df = self.mask_data_cache[contour_csv_path]
        slice_rows = df[df["SliceID"] == slice_id]
    
        label_map = np.zeros((height, width), dtype=np.int64)
    
        for _, row in slice_rows.iterrows():
            encoded_pixels = row["EncodedPixels"]
    
            if str(encoded_pixels) != "-1":
                mask_2d = pixel_decoder(encoded_pixels, height, width)
                organ_id = int(row["MaskTypeID"]) + 1
                label_map[mask_2d == 1] = organ_id
    
        img = tv_tensors.Image(torch.from_numpy(img).unsqueeze(0))
        target = tv_tensors.Mask(torch.from_numpy(label_map))
    
        if self.transform is not None:
            img, target = self.transform(img, target)
    
        img = img.as_subclass(torch.Tensor).float()
        target = target.as_subclass(torch.Tensor).long()
    
        return img, target


def collect_slice_pairs(dataset_path):
    slice_contour_pairs = []
    dataset_cases = sorted(Path(dataset_path).iterdir())

    for case_path in dataset_cases:
        if not case_path.is_dir():
            continue

        case_days = sorted(case_path.iterdir())
        for case_day in case_days:
            scans_path = case_day / "scans"
            contours_path = case_day / "contours"
            contour_csv_path = contours_path / "masks_rle.csv"

            if not scans_path.is_dir() or not contour_csv_path.is_file():
                continue

            for scan_path in sorted(scans_path.iterdir()):
                if not scan_path.is_file():
                    continue
                slice_parts = scan_path.name.split("_")
                if len(slice_parts) < 2:
                    continue
                slice_id = slice_parts[0] + "_" + slice_parts[1]
                slice_contour_pairs.append((slice_id, scan_path, contour_csv_path))

    if not slice_contour_pairs:
        raise ValueError(f"No scan/mask pairs found inside: {dataset_path}")

    return slice_contour_pairs


def split_pairs_by_scan(slice_contour_pairs, seed):
    import random

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
