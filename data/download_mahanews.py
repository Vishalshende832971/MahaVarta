"""
Download the official L3Cube-MahaNews (LPC) dataset.
Source: https://github.com/l3cube-pune/MarathiNLP/tree/main/L3Cube-MahaNews/LPC
"""

import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "mahanews")
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    "train": "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaNews/LPC/LPC_Train.csv",
    "valid": "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaNews/LPC/LPC_Valid.csv",
    "test": "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaNews/LPC/LPC_Test.csv",
}


def download_and_verify():
    stats = {}
    for split, url in URLS.items():
        file_path = os.path.join(DATA_DIR, f"{split}.csv")
        if not os.path.exists(file_path):
            print(f"Downloading {split} from {url}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Saved {file_path}")
        else:
            print(f"Already downloaded: {file_path}")

        df = pd.read_csv(file_path)
        stats[split] = {
            "rows": len(df),
            "columns": list(df.columns),
            "null_texts": int(df["Text"].isnull().sum()),
            "null_labels": int(df["Label"].isnull().sum()),
            "unique_labels": sorted(df["Label"].dropna().unique().tolist()),
            "num_classes": int(df["Label"].nunique()),
        }

    return stats


if __name__ == "__main__":
    stats = download_and_verify()
    for split, s in stats.items():
        print(f"\n--- {split.upper()} ---")
        print(f"Rows: {s['rows']}")
        print(f"Columns: {s['columns']}")
        print(f"Null texts: {s['null_texts']}, Null labels: {s['null_labels']}")
        print(f"Num classes: {s['num_classes']}")
        print(f"Classes: {s['unique_labels']}")
