import argparse
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from download_dataset import download_dataset
from project_utils import project_path

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_TICKETS_PER_QUEUE = 5


def normalize_column_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r"\s+", "_", str(column).strip().lower())
        for column in df.columns
    ]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the support ticket dataset."
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Download a fresh copy instead of reusing data/raw/customer_support_tickets.csv.",
    )
    args = parser.parse_args()

    raw_path = download_dataset(force=args.redownload)
    print("\nReading:", raw_path)

    df = pd.read_csv(raw_path, low_memory=False)
    df = normalize_column_names(df)

    required = {"subject", "body", "queue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"The selected source file is missing required columns: {sorted(missing)}"
        )

    optional = [
        column for column in ("priority", "type", "language")
        if column in df.columns
    ]
    keep_columns = ["subject", "body", "queue"] + optional
    df = df[keep_columns].copy()

    for column in ("subject", "body", "queue"):
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[df["queue"].ne("")].copy()
    df["text"] = (
        df["subject"].str.strip() + " " + df["body"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].ne("")].copy()

    rows_before_dedup = len(df)
    df["normalized_text"] = df["text"].str.casefold()
    # Remove exact duplicate ticket texts before splitting to reduce leakage.
    df = df.drop_duplicates(subset=["normalized_text"], keep="first")
    df = df.drop(columns=["normalized_text"]).reset_index(drop=True)
    duplicates_removed = rows_before_dedup - len(df)

    counts = df["queue"].value_counts()
    rare_queues = counts[counts < MIN_TICKETS_PER_QUEUE].index.tolist()
    rare_rows_removed = int(df["queue"].isin(rare_queues).sum())
    if rare_queues:
        df = df[~df["queue"].isin(rare_queues)].copy()

    if df["queue"].nunique() < 2:
        raise ValueError("Not enough queue classes remain after cleaning.")

    # A stratified 80/20 split needs enough records for every queue to appear
    # in both partitions. MIN_TICKETS_PER_QUEUE protects rare classes.
    class_count = df["queue"].nunique()
    test_count = int(round(len(df) * TEST_SIZE))
    train_count = len(df) - test_count
    if test_count < class_count or train_count < class_count:
        raise ValueError(
            "There are too few rows to stratify this dataset into train/test sets. "
            "Keep more data or lower the number of queue classes."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["queue"],
    )

    data_dir = project_path("data")
    results_dir = project_path("results")
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(data_dir / "cleaned_tickets.csv", index=False)
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    try:
        with project_path("data", "raw", "source_info.json").open(
            "r", encoding="utf-8"
        ) as file:
            source_info = json.load(file)
    except FileNotFoundError:
        source_info = {"source": "unknown"}

    info = {
        **source_info,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "source_rows_loaded": int(rows_before_dedup),
        "rows_after_deduplication_and_filtering": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "queue_classes": int(df["queue"].nunique()),
        "duplicate_texts_removed": int(duplicates_removed),
        "rows_removed_for_rare_queues": int(rare_rows_removed),
        "minimum_tickets_per_queue": MIN_TICKETS_PER_QUEUE,
        "target": "queue",
        "input_text": "subject + body",
    }
    with (results_dir / "dataset_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=4, ensure_ascii=False)

    print("\nDataset preparation completed.")
    print(f"Raw CSV: {raw_path}")
    print(f"Cleaned CSV: {data_dir / 'cleaned_tickets.csv'}")
    print(f"Training CSV: {data_dir / 'train.csv'}")
    print(f"Testing CSV: {data_dir / 'test.csv'}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(f"Queue classes: {df['queue'].nunique():,}")
    print(f"Duplicate texts removed: {duplicates_removed:,}")
    print(f"Rare-queue rows removed: {rare_rows_removed:,}")


if __name__ == "__main__":
    main()
