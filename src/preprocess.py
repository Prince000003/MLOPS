"""Clean the raw data and create train/test CSV files."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "data.csv")
PROCESSED = os.path.join(BASE, "data", "processed")


def preprocess(raw_path=RAW, out_dir=PROCESSED, test_size=0.2, seed=42):
    df = pd.read_csv(raw_path)
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["target"]
    )

    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"Preprocessed: {len(train)} train rows, {len(test)} test rows")
    return train, test


if __name__ == "__main__":
    preprocess()
