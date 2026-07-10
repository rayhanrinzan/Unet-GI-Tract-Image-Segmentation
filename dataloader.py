from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


IMAGE_SIZE = 266

# 0 = background
# 1 = large_bowel
# 2 = small_bowel
# 3 = stomach
NUM_CLASSES = 4

EVAL_CLASS_MAP = {
    1: 1,  # MaskTypeID 1 = large_bowel
    3: 2,  # MaskTypeID 3 = small_bowel
    4: 3,  # MaskTypeID 4 = stomach
}


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
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomVerticalFlip(p=0.3),
])

eval_transform = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
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
            mask_type_id = int(row["MaskTypeID"])
            encoded_pixels = row["EncodedPixels"]
        
            # Ignore ampulla_of_vater and pyloric_sphincter for now.
            # Keep only large_bowel, small_bowel, stomach.
            if mask_type_id not in EVAL_CLASS_MAP:
                continue
        
            if pd.isna(encoded_pixels) or str(encoded_pixels) == "-1":
                continue
        
            mask_2d = pixel_decoder(encoded_pixels, height, width)
            class_id = EVAL_CLASS_MAP[mask_type_id]
            label_map[mask_2d == 1] = class_id

    
        img = tv_tensors.Image(torch.from_numpy(img).unsqueeze(0))
        target = tv_tensors.Mask(torch.from_numpy(label_map))
    
        if self.transform is not None:
            img, target = self.transform(img, target)
    
        img = img.as_subclass(torch.Tensor).float()
        target = target.as_subclass(torch.Tensor).long()
    
        return img, target


class PrecomputedMaskDataset(Dataset):
    def __init__(self, image_mask_pairs, transform=None):
        self.image_mask_pairs = image_mask_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]

        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        mask = np.load(mask_path)
        if mask.ndim == 3:
            label_map = np.zeros(mask.shape[:2], dtype=np.int64)
            label_map[mask[..., 0] > 0] = 1
            label_map[mask[..., 1] > 0] = 2
            label_map[mask[..., 2] > 0] = 3
        elif mask.ndim == 2:
            label_map = mask.astype(np.int64)
        else:
            raise ValueError(f"Unexpected mask shape for {mask_path}: {mask.shape}")

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


def resolve_mask_dataset_path(mask_root_path):
    if not mask_root_path:
        raise ValueError(
            "No precomputed mask dataset path provided. Set UWMGI_MASK_DATASET_PATH "
            "or pass --mask-dataset-root."
        )

    mask_root = Path(mask_root_path).expanduser().resolve()
    if not (mask_root / "train.csv").is_file():
        raise ValueError(f"Could not find train.csv inside precomputed mask dataset: {mask_root}")

    return mask_root


def _localize_kaggle_path(path_value, original_dataset_path, mask_dataset_path):
    path = Path(str(path_value))

    if path.is_absolute() and path.exists():
        return path

    parts = path.parts
    if "uw-madison-gi-tract-image-segmentation" in parts:
        idx = parts.index("uw-madison-gi-tract-image-segmentation")
        relative = Path(*parts[idx + 1:])
        candidates = [
            original_dataset_path.parent / relative,
            original_dataset_path / relative,
        ]
        if relative.parts and relative.parts[0] == "train":
            no_train_relative = Path(*relative.parts[1:])
            candidates.extend([
                original_dataset_path.parent / no_train_relative,
                original_dataset_path / no_train_relative,
            ])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        slice_parts = relative.name.split("_")
        if len(slice_parts) >= 2:
            slice_prefix = f"{slice_parts[0]}_{slice_parts[1]}_"
            for candidate in candidates:
                if candidate.parent.is_dir():
                    matches = sorted(candidate.parent.glob(f"{slice_prefix}*{candidate.suffix}"))
                    if matches:
                        return matches[0]

    if "uwmgi-mask-dataset" in parts:
        idx = parts.index("uwmgi-mask-dataset")
        candidate = mask_dataset_path / Path(*parts[idx + 1:])
        if candidate.exists():
            return candidate

    candidate = mask_dataset_path / path
    if candidate.exists():
        return candidate

    candidate = original_dataset_path / path
    if candidate.exists():
        return candidate

    return path


def collect_precomputed_mask_pairs(original_dataset_path, mask_dataset_path):
    original_dataset_path = Path(original_dataset_path).expanduser().resolve()
    mask_dataset_path = resolve_mask_dataset_path(mask_dataset_path)

    df = pd.read_csv(mask_dataset_path / "train.csv")
    df = df.groupby("id").head(1).reset_index(drop=True)
    df["mask_path"] = df["mask_path"].str.replace("/png/", "/np/", regex=False)
    df["mask_path"] = df["mask_path"].str.replace(".png", ".npy", regex=False)

    image_mask_pairs = []
    for _, row in df.iterrows():
        image_path = _localize_kaggle_path(row["image_path"], original_dataset_path, mask_dataset_path)
        mask_path = _localize_kaggle_path(row["mask_path"], original_dataset_path, mask_dataset_path)

        if not image_path.is_file():
            raise FileNotFoundError(f"Could not find image from mask train.csv: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Could not find precomputed mask from train.csv: {mask_path}")

        image_mask_pairs.append((image_path, mask_path))

    if not image_mask_pairs:
        raise ValueError(f"No image/mask pairs found from: {mask_dataset_path / 'train.csv'}")

    return image_mask_pairs


def split_pairs_by_scan(slice_contour_pairs, seed):
    import random

    unique_scan_ids = sorted({pair[1].parent.parent.name for pair in slice_contour_pairs})
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


def split_precomputed_pairs_by_scan(image_mask_pairs, seed):
    import random

    unique_scan_ids = sorted({pair[0].parent.parent.name for pair in image_mask_pairs})
    random.seed(seed)
    random.shuffle(unique_scan_ids)

    num_scans = len(unique_scan_ids)
    train_idx = int(0.7 * num_scans)
    val_idx = train_idx + int(0.1 * num_scans)

    train_scan_ids = set(unique_scan_ids[:train_idx])
    val_scan_ids = set(unique_scan_ids[train_idx:val_idx])
    test_scan_ids = set(unique_scan_ids[val_idx:])

    train_pairs, val_pairs, test_pairs = [], [], []
    for pair in image_mask_pairs:
        scan_id = pair[0].parent.parent.name
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
