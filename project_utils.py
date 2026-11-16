from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)


def load_data():
    train_path = project_path("data", "train.csv")
    test_path = project_path("data", "test.csv")

    missing = [path for path in (train_path, test_path) if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Dataset split files are missing:\n"
            f"{missing_text}\nRun `python prepare_data.py` first."
        )

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    for name, frame in (("train.csv", train_df), ("test.csv", test_df)):
        required = {"text", "queue"}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing_columns)}. "
                "Run prepare_data.py again."
            )

        frame["text"] = frame["text"].fillna("").astype(str).str.strip()
        frame["queue"] = frame["queue"].fillna("").astype(str).str.strip()
        frame.drop(
            frame.index[frame["text"].eq("") | frame["queue"].eq("")],
            inplace=True,
        )
        frame.reset_index(drop=True, inplace=True)

        if frame.empty:
            raise ValueError(f"{name} has no usable rows.")

    if train_df["queue"].nunique() < 2:
        raise ValueError("Training data must contain at least two queue classes.")

    unseen = set(test_df["queue"].unique()) - set(train_df["queue"].unique())
    if unseen:
        raise ValueError(
            f"Test set has queue labels absent from train set: {sorted(unseen)}"
        )

    train_texts = set(train_df["text"].str.casefold())
    test_texts = set(test_df["text"].str.casefold())
    overlap = len(train_texts.intersection(test_texts))
    if overlap:
        print(
            f"WARNING: {overlap} identical texts occur in both splits. "
            "Check data leakage before reporting the results."
        )

    return train_df, test_df
