"""
Final evaluation against held-out test.csv after training.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PlateDataset, ctc_collate, clean_plate_text
from model import CNNGRUCTC, greedy_decode


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def main():
    script_dir = Path(__file__).resolve().parent

    def resolve_path(path_str):
        path = Path(path_str)
        if path.is_absolute():
            return path
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return (script_dir / path).resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', default='test.csv')
    parser.add_argument('--weights', default='checkpoints/best.pth')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    args.test_csv = str(resolve_path(args.test_csv))
    args.weights = str(resolve_path(args.weights))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weights_path}')

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = CNNGRUCTC().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {checkpoint.get('epoch')}  "
          f"val_exact={checkpoint.get('val_accuracy')}  "
          f"input={checkpoint.get('image_width')}x{checkpoint.get('image_height')}")

    test_ds = PlateDataset(args.test_csv, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate
    )

    exact_correct = 0
    total = 0
    total_char_errors = 0
    total_chars = 0

    print('=' * 60)
    print('FINAL TEST — held-out test.csv')
    print('=' * 60)

    with torch.no_grad():
        for images, targets, target_lengths, texts in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = greedy_decode(logits.cpu())
            for pred, actual in zip(preds, texts):
                actual = clean_plate_text(actual)
                is_correct = pred == actual
                exact_correct += int(is_correct)
                total += 1
                dist = edit_distance(pred, actual)
                total_char_errors += dist
                total_chars += len(actual)
                print(
                    f'Actual: {actual:12s} Predicted: {pred:12s} '
                    f'{"CORRECT" if is_correct else "WRONG"} '
                    f'(char edits: {dist}/{len(actual)})'
                )

    exact_accuracy = (exact_correct / total) * 100 if total else 0
    cer = (total_char_errors / total_chars) * 100 if total_chars else 0
    char_accuracy = 100 - cer
    print()
    print('=' * 60)
    print('FINAL RESULT')
    print('=' * 60)
    print(f'Exact full-plate match : {exact_correct}/{total}  ({exact_accuracy:.2f}%)')
    print(f'Character Error Rate    : {cer:.2f}%')
    print(f'Character-level accuracy: {char_accuracy:.2f}%')


if __name__ == '__main__':
    main()
