import argparse
import random
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader

from dataset import PlateDataset, SyntheticPlateDataset, ctc_collate, clean_plate_text
from model import (
    ALPHABET,
    BLANK,
    CNNGRUCTC,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    greedy_decode,
)


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

    parser = argparse.ArgumentParser(
        description='Train CNN+BiGRU+CTC plate recognizer (Shi & Zhao 2023, Indian plates)'
    )
    parser.add_argument('--train-csv', default='train.csv')
    parser.add_argument('--val-csv', default='val.csv')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Adam LR. Paper used 0.01 with batch 128; 0.001 is safer on CPU/small batch.')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--out', default='checkpoints')
    parser.add_argument('--synth', type=int, default=1500,
                        help='On-the-fly synthetic Indian plates mixed into training. 0 disables.')
    parser.add_argument('--overfit', type=int, default=0,
                        help='If >0, train only on this many real samples to verify the pipeline.')
    args = parser.parse_args()

    args.train_csv = str(resolve_path(args.train_csv))
    args.val_csv = str(resolve_path(args.val_csv))
    args.out = str(resolve_path(args.out))

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print(f'Input={IMAGE_WIDTH}x{IMAGE_HEIGHT}  Epochs={args.epochs}  '
          f'Batch={args.batch_size}  LR={args.lr}  Synth={args.synth}')

    train_ds = PlateDataset(args.train_csv, augment=True)
    val_ds = PlateDataset(args.val_csv, augment=False)
    if args.overfit:
        train_ds.rows = train_ds.rows[:args.overfit]
        val_ds.rows = train_ds.rows
        print(f'OVERFIT mode: {len(train_ds)} samples')
    if args.synth > 0 and not args.overfit:
        train_ds = ConcatDataset([train_ds, SyntheticPlateDataset(size=args.synth, augment=True)])
        print(f'Real+synth training images per epoch: {len(train_ds)}')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=ctc_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=ctc_collate,
    )

    model = CNNGRUCTC().to(device)
    criterion = CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Paper Section IV-C2: Adam + dynamic learning-rate decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-5
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_char = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            optimizer.zero_grad()
            logits = model(images)
            log_probs = logits.log_softmax(2)
            input_lengths = torch.full(
                (images.size(0),), logits.size(0), dtype=torch.long, device=device
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        exact = 0
        total = 0
        char_err = 0
        char_total = 0
        sample_pairs = []
        with torch.no_grad():
            for images, targets, target_lengths, texts in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                logits = model(images)
                log_probs = logits.log_softmax(2)
                input_lengths = torch.full(
                    (images.size(0),), logits.size(0), dtype=torch.long, device=device
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                if torch.isfinite(loss):
                    val_loss_sum += loss.item()
                    val_batches += 1
                preds = greedy_decode(logits.cpu())
                gts = [clean_plate_text(t) for t in texts]
                for pred, gt in zip(preds, gts):
                    exact += int(pred == gt)
                    total += 1
                    char_err += edit_distance(pred, gt)
                    char_total += max(len(gt), 1)
                    if len(sample_pairs) < 4:
                        sample_pairs.append((gt, pred))

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        accuracy = 100.0 * exact / max(total, 1)
        char_acc = 100.0 * (1.0 - char_err / max(char_total, 1))
        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step(char_acc)

        print(
            f'Epoch {epoch:03d}/{args.epochs} | lr={cur_lr:.6f} | '
            f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | '
            f'val_exact={accuracy:.2f}% | val_char={char_acc:.2f}%'
        )
        for gt, pred in sample_pairs:
            print(f'  GT={gt:12s} PRED={pred}')

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': accuracy,
            'val_char_acc': char_acc,
            'alphabet': ALPHABET,
            'blank_index': BLANK,
            'image_height': IMAGE_HEIGHT,
            'image_width': IMAGE_WIDTH,
        }
        torch.save(checkpoint, out_dir / 'latest.pth')
        is_best = (accuracy > best_acc) or (accuracy == best_acc and char_acc > best_char)
        if is_best:
            best_acc = accuracy
            best_char = char_acc
            torch.save(checkpoint, out_dir / 'best.pth')
            print(f'  Saved best checkpoint (exact={accuracy:.2f}%, char={char_acc:.2f}%).')

    print(f'\nTraining finished. Best checkpoint: {out_dir / "best.pth"}')
    print('Evaluate with: python test_recognition.py --weights checkpoints/best.pth')


if __name__ == '__main__':
    main()
