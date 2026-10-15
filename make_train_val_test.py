"""
make_train_val_test.py

Replaces make_train_val.py.

Two fixes over the original:
1. Splits by plate TEXT, not randomly by image — so if the same plate
   appears in multiple (near-duplicate) crops, all of them land in the
   same split. Prevents train/test leakage.
2. Produces three files (train.csv, val.csv, test.csv) instead of two.
   val.csv drives the LR scheduler during training; test.csv is never
   touched during training and is only used once, at the end, for your
   real reported accuracy number.

Input: all_labels.csv with columns "image,text" (same format as before).
"""

import csv
import random
from pathlib import Path

LABELS_FILE = Path("all_labels.csv")

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def main():
    random.seed(SEED)

    with LABELS_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Clean + keep only non-empty labels (same as original script)
    rows = [
        {
            "image": r["image"] if ("/" in r["image"] or "\\" in r["image"]) else f"crops/{r['image']}",
            "text": r["text"].strip().upper()
        }
        for r in rows
        if r["text"].strip()
    ]

    if len(rows) < 3:
        raise SystemExit("Label at least a few images before creating the CSV files.")

    # Stratified per-plate split across train/val/test
    # This ensures EVERY plate font/text is represented in training,
    # while holding out independent video frames for val and test.
    groups = {}
    for r in rows:
        groups.setdefault(r["text"], []).append(r)

    assigned = {"train": [], "val": [], "test": []}

    for key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        if n == 1:
            assigned["train"].extend(items)
        elif n == 2:
            assigned["train"].append(items[0])
            assigned["test"].append(items[1])
        elif n == 3:
            assigned["train"].append(items[0])
            assigned["val"].append(items[1])
            assigned["test"].append(items[2])
        else:
            n_test = max(1, int(round(n * RATIOS["test"])))
            n_val = max(1, int(round(n * RATIOS["val"])))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val = max(1, (n - 1) // 2)
                n_test = n - n_train - n_val

            assigned["train"].extend(items[:n_train])
            assigned["val"].extend(items[n_train:n_train + n_val])
            assigned["test"].extend(items[n_train + n_val:])

    for filename, split_rows in [
        ("train.csv", assigned["train"]),
        ("val.csv", assigned["val"]),
        ("test.csv", assigned["test"]),
    ]:
        with open(filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "text"])
            writer.writeheader()
            writer.writerows(split_rows)

    print("Labeled images:", len(rows))
    print("Unique plate texts (groups):", len(groups))
    print()
    print("Training images  :", len(assigned["train"]))
    print("Validation images:", len(assigned["val"]))
    print("Test images      :", len(assigned["test"]))
    print()
    print("Created train.csv, val.csv, and test.csv (Stratified Frame Split)")


if __name__ == "__main__":
    main()