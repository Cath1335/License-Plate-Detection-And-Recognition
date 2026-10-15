import torch
import torch.nn as nn
import torch.nn.functional as F

# CTC blank is index 0. Do not use '-' in ground-truth labels.
CHARS = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
BLANK = 0
NUM_CLASSES = len(CHARS)

# Paper Fig.4 CNN expects a short, wide plate. Height 32 after pooling
# reaches 1; width 192 keeps ~48 CTC time-steps for 8-10 Indian characters.
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 192


class CNNGRUCTC(nn.Module):
    """
    Recognition network from Shi & Zhao, IEEE Access 2023, Section III-D / Fig.4:
    7-layer CNN -> 3x3 sliding-window sequence features -> BiGRU -> CTC classes.

    Indian plates use 36 alphanumeric classes + blank instead of the paper's
    7-character Chinese vocabulary; CTC still handles variable length (8-10).
    """

    def __init__(self, hidden_size=256, gru_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Paper: 3x3 window around each conv5 point -> 9C features, then project.
        self.slide_proj = nn.Sequential(
            nn.Conv2d(512 * 9, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.sequence = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.2 if gru_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        features = self.cnn(x)
        features = F.adaptive_avg_pool2d(features, (1, features.size(3)))
        windows = F.unfold(features, kernel_size=3, padding=1)
        b, _, n = windows.shape
        windows = windows.view(b, 512 * 9, 1, n)
        features = self.slide_proj(windows).squeeze(2).permute(2, 0, 1)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)


def greedy_decode(logits):
    """CTC greedy decode: argmax, collapse repeats, drop blanks."""
    best = logits.argmax(dim=2)
    results = []
    for b in range(best.shape[1]):
        out = []
        previous = None
        for idx in best[:, b].tolist():
            if idx != previous and idx != BLANK:
                out.append(CHARS[idx])
            previous = idx
        results.append("".join(out))
    return results
