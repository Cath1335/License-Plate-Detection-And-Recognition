
import csv
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV_FILE = BASE / "train.csv"
PREDICT_FILE = BASE / "predict.py"
WEIGHTS = BASE / "checkpoints" / "best.pth"

with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

correct = 0

for row in rows:
    image = row["image"]
    actual = row["text"].strip().upper()

    result = subprocess.run(
        [
            sys.executable,
            str(PREDICT_FILE),
            "--image", str(BASE / image),
            "--weights", str(WEIGHTS),
        ],
        capture_output=True,
        text=True,
        cwd=BASE,
    )

    output = result.stdout

    if result.returncode != 0:
        print(f"\nERROR: {image}")
        print(result.stderr)
        continue

    predicted = ""
    for line in output.splitlines():
        if "Predicted plate text" in line:
            predicted = line.split(":", 1)[1].strip()
            break

    is_correct = predicted == actual
    correct += int(is_correct)

    print(f"Image: {image}")
    print(f"Actual: {actual} | Predicted: {predicted}")
    print("Correct" if is_correct else "Incorrect")
    print("-" * 45)

print(f"\nTraining-set exact matches: {correct}/{len(rows)}")
