import argparse
from pathlib import Path

import kagglehub


def main():
    """Download the dataset once and persist the resolved local path to a text file."""
    parser = argparse.ArgumentParser(description="Download the UW-Madison GI Tract dataset once.")
    parser.add_argument(
        "--path-file",
        default="dataset_path.txt",
        help="File used to store the downloaded dataset path.",
    )
    args = parser.parse_args()

    dataset_path = kagglehub.dataset_download("happyharrycn/uw-madison-gi-tract-image-segmentation-dataset")
    Path(args.path_file).write_text(dataset_path + "\n", encoding="utf-8")

    print("Dataset downloaded to:", dataset_path)
    print("Saved dataset path to:", args.path_file)
    print("Set env var before training:")
    print(f'  export GI_TRACT_DATASET_PATH="{dataset_path}"')


if __name__ == "__main__":
    main()
