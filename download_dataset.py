import json
import shutil
from pathlib import Path

import pandas as pd

from project_utils import project_path

HF_DATASET_ID = "Tobi-Bueck/customer-support-tickets"
HF_PRIMARY_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
KAGGLE_DATASET_ID = "tobiasbueck/multilingual-customer-support-tickets"


def _has_required_columns(frame):
    columns = {str(column).strip().lower() for column in frame.columns}
    return {"subject", "body", "queue"}.issubset(columns)


def _save_source_frame(frame, raw_path, source_name, source_file):
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)

    info = {
        "source": source_name,
        "source_file": source_file,
        "rows_downloaded": int(len(frame)),
        "local_file": str(raw_path),
        "required_fields": ["subject", "body", "queue"],
    }
    with project_path("data", "raw", "source_info.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(info, file, indent=4)

    print(f"Dataset saved to: {raw_path}")
    print(f"Rows downloaded: {len(frame):,}")
    print(f"Source: {source_name}")
    return raw_path


def _download_from_huggingface(raw_path):
    # The public dataset loader normally provides the full viewer split.
    # If it fails, try downloading the named current multilingual CSV directly.
    try:
        from datasets import load_dataset

        print(f"Trying Hugging Face dataset: {HF_DATASET_ID}")
        dataset = load_dataset(HF_DATASET_ID, split="train")
        frame = dataset.to_pandas()

        if not _has_required_columns(frame):
            raise ValueError(
                "The dataset viewer did not return subject/body/queue columns."
            )

        return _save_source_frame(
            frame,
            raw_path,
            f"Hugging Face: {HF_DATASET_ID}",
            "train split loaded by datasets",
        )
    except Exception as exc:
        print(f"Full Hugging Face dataset load failed: {exc}")

    try:
        from huggingface_hub import hf_hub_download

        print(f"Trying the main multilingual CSV: {HF_PRIMARY_FILE}")
        cached_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=HF_PRIMARY_FILE,
            repo_type="dataset",
        )
        frame = pd.read_csv(cached_path, low_memory=False)
        if not _has_required_columns(frame):
            raise ValueError(
                f"{HF_PRIMARY_FILE} does not contain subject/body/queue."
            )

        return _save_source_frame(
            frame,
            raw_path,
            f"Hugging Face: {HF_DATASET_ID}",
            HF_PRIMARY_FILE,
        )
    except Exception as exc:
        print(f"Direct Hugging Face CSV download failed: {exc}")
        return None


def _download_from_kaggle(raw_path):
    try:
        import glob
        import os
        import kagglehub

        print(f"Trying KaggleHub dataset: {KAGGLE_DATASET_ID}")
        downloaded_dir = kagglehub.dataset_download(KAGGLE_DATASET_ID)
        csv_paths = glob.glob(
            os.path.join(downloaded_dir, "**", "*.csv"),
            recursive=True,
        )

        # Prefer the current multilingual version if the Kaggle folder has it.
        preferred = [
            path for path in csv_paths
            if "multi-lang-5-2-50-version" in Path(path).name.lower()
        ]
        candidates = preferred or csv_paths

        best_frame = None
        best_path = None
        best_rows = -1

        for csv_path in candidates:
            try:
                frame = pd.read_csv(csv_path, low_memory=False)
                if _has_required_columns(frame) and len(frame) > best_rows:
                    best_frame = frame
                    best_path = csv_path
                    best_rows = len(frame)
            except Exception as exc:
                print(f"Skipping CSV {csv_path}: {exc}")

        if best_frame is None:
            raise FileNotFoundError(
                "KaggleHub downloaded the dataset, but no compatible CSV was found."
            )

        return _save_source_frame(
            best_frame,
            raw_path,
            f"KaggleHub: {KAGGLE_DATASET_ID}",
            str(best_path),
        )
    except Exception as exc:
        print(f"KaggleHub download failed: {exc}")
        return None


def download_dataset(force=False):
    """
    Download the chosen support-ticket dataset.

    The CSV is stored at data/raw/customer_support_tickets.csv.
    Existing local data is reused unless force=True.
    """
    raw_path = project_path("data", "raw", "customer_support_tickets.csv")

    if raw_path.is_file() and not force:
        print(f"Using existing dataset file: {raw_path}")
        return raw_path

    result = _download_from_huggingface(raw_path)
    if result is not None:
        return result

    result = _download_from_kaggle(raw_path)
    if result is not None:
        return result

    raise RuntimeError(
        "Could not download the support-ticket dataset from Hugging Face or KaggleHub. "
        "Check your internet connection, then retry. As a manual fallback, download "
        f"{HF_PRIMARY_FILE} from https://huggingface.co/datasets/{HF_DATASET_ID}/tree/main "
        f"and save it as {raw_path}."
    )


if __name__ == "__main__":
    download_dataset(force=True)
