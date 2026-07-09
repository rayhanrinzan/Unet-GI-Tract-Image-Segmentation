#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

CLASS_MAP = {
    1: 1,  # large_bowel
    3: 2,  # small_bowel
    4: 3,  # stomach
}


def resolve_dataset_dir(dataset_root: Path) -> Path:
    if dataset_root.name == "dataset" and dataset_root.is_dir():
        return dataset_root
    nested = dataset_root / "dataset"
    if nested.is_dir():
        return nested
    raise ValueError(
        f"Could not find dataset directory. Expected either '{dataset_root}' "
        f"or '{nested}' to exist."
    )


def find_rle_csv(contours_dir: Path) -> Path | None:
    for name in ("masks_rle.csv", "mask_rle.csv"):
        candidate = contours_dir / name
        if candidate.is_file():
            return candidate
    return None


def rle_decode(encoded_pixels: str, height: int, width: int) -> np.ndarray:
    parts = str(encoded_pixels).split()
    if not parts:
        return np.zeros((height, width), dtype=np.uint8)

    starts = np.asarray(parts[0::2], dtype=np.int64) - 1
    lengths = np.asarray(parts[1::2], dtype=np.int64)
    ends = starts + lengths

    flat = np.zeros(height * width, dtype=np.uint8)
    for start, end in zip(starts, ends):
        flat[start:end] = 1

    return flat.reshape((height, width))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build segx metadata CSV and multiclass GI masks from UW GI RLE labels."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to dataset/ directory or its parent.",
    )
    parser.add_argument("--out-csv", default="metadata/train.csv")
    parser.add_argument("--out-mask-dir", default="metadata/masks")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_dir = resolve_dataset_dir(dataset_root)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_mask_dir = Path(args.out_mask_dir).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    case_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    for case_dir in tqdm(case_dirs, desc="Cases"):
        day_dirs = sorted(path for path in case_dir.iterdir() if path.is_dir())
        for day_dir in day_dirs:
            scans_dir = day_dir / "scans"
            contours_dir = day_dir / "contours"
            rle_csv = find_rle_csv(contours_dir)
            if not scans_dir.is_dir() or rle_csv is None:
                continue

            rle_df = pd.read_csv(rle_csv)

            for scan_path in sorted(path for path in scans_dir.iterdir() if path.is_file()):
                name_parts = scan_path.name.split("_")
                if len(name_parts) < 2:
                    continue

                slice_id = f"{name_parts[0]}_{name_parts[1]}"
                image = cv2.imread(str(scan_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue

                height, width = image.shape[:2]
                label_map = np.zeros((height, width), dtype=np.uint8)

                slice_rows = rle_df[rle_df["SliceID"] == slice_id]
                for _, row in slice_rows.iterrows():
                    mask_type_id = int(row["MaskTypeID"])
                    encoded_pixels = row["EncodedPixels"]

                    class_id = CLASS_MAP.get(mask_type_id)
                    if class_id is None:
                        continue
                    if pd.isna(encoded_pixels) or str(encoded_pixels) == "-1":
                        continue

                    mask = rle_decode(str(encoded_pixels), height, width)
                    label_map[mask == 1] = class_id

                sample_id = f"{case_dir.name}_{day_dir.name}_{scan_path.stem}"
                mask_path = out_mask_dir / f"{sample_id}.png"
                if not cv2.imwrite(str(mask_path), label_map):
                    raise RuntimeError(f"Failed to write mask: {mask_path}")

                rows.append(
                    {
                        "ID": sample_id,
                        "FNAME": str(scan_path.resolve()),
                        "MASK_gi_organs": str(mask_path.resolve()),
                    }
                )

    if not rows:
        raise ValueError(f"No training rows generated from dataset at {dataset_dir}")

    metadata_df = pd.DataFrame(rows, columns=["ID", "FNAME", "MASK_gi_organs"])
    metadata_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(metadata_df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
