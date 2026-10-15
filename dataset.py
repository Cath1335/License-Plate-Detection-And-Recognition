import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import ALPHABET, CHAR_TO_IDX, IMAGE_HEIGHT, IMAGE_WIDTH

STATES = [
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DL', 'DN', 'GA', 'GJ', 'HP',
    'HR', 'JH', 'JK', 'KA', 'KL', 'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD',
    'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB',
]
SERIES_LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'


def clean_plate_text(text):
    return "".join(c for c in str(text).upper() if c in ALPHABET)


def flatten_two_line_plate(image):
    """
    Indian plates are often two rows (state+district on top, series+number below).
    The paper's CNN+GRU assumes a single left-to-right sequence, so stack the
    two rows horizontally before the sequence model sees the crop.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    if width / height >= 2.3:
        return image
    mid = height // 2
    top = image.crop((0, 0, width, mid))
    bottom = image.crop((0, mid, width, height))
    merged = Image.new(image.mode, (width * 2, mid), (255, 255, 255))
    merged.paste(top, (0, 0))
    merged.paste(bottom, (width, 0))
    return merged


def resize_keep_width(image, height=IMAGE_HEIGHT, max_width=IMAGE_WIDTH):
    """Scale to fixed height, keep aspect ratio, pad on the right (CRNN-style)."""
    image = image.convert('RGB')
    orig_w, orig_h = image.size
    if orig_h <= 0:
        orig_h = 1
    new_w = max(8, int(round(orig_w * height / orig_h)))
    new_w = min(new_w, max_width)
    image = image.resize((new_w, height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (max_width, height), (128, 128, 128))
    canvas.paste(image, (0, 0))
    return canvas


def preprocess_plate_image(image):
    """
    Standardize plate crop:
    1. Flatten two-line Indian plates into one sequence
    2. Grayscale + contrast stretch (yellow taxi vs white private, IND badge)
    3. Keep-ratio resize to 32x192
    """
    image = flatten_two_line_plate(image.convert('RGB'))
    gray = image.convert('L')
    enhanced = ImageOps.autocontrast(gray, cutoff=1)
    padded = ImageOps.expand(enhanced, border=(3, 2), fill=255)
    padded = resize_keep_width(padded.convert('RGB'))
    return padded


def _load_font(size):
    for path in (
        r'C:\Windows\Fonts\arialbd.ttf',
        r'C:\Windows\Fonts\consolab.ttf',
        r'C:\Windows\Fonts\courbd.ttf',
        r'C:\Windows\Fonts\arial.ttf',
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_synthetic_plate(text):
    """Render a single-line Indian plate used to teach character shapes."""
    yellow = random.random() < 0.25
    bg = (255, 206, 0) if yellow else (238, 238, 238)
    fg = (20, 20, 20)
    img = Image.new('RGB', (256, 64), bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle((2, 2, 253, 61), outline=(25, 25, 25), width=3)
    font = _load_font(random.choice([26, 28, 30, 32]))
    spaced = ' '.join((text[:2], text[2:4], text[4:-4], text[-4:])) if len(text) >= 8 else text
    bbox = draw.textbbox((0, 0), spaced, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(8, (256 - tw) // 2)
    y = max(4, (64 - th) // 2 - 2)
    draw.text((x, y), spaced, font=font, fill=fg)
    if random.random() < 0.4:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.25))
    if random.random() < 0.3:
        img = img.resize((128, 32), Image.Resampling.BILINEAR).resize((256, 64), Image.Resampling.BILINEAR)
    return img


def random_indian_plate_text():
    state = random.choice(STATES)
    district = f'{random.randint(1, 99):02d}'
    series = ''.join(random.choices(SERIES_LETTERS, k=random.choice([1, 2])))
    number = f'{random.randint(0, 9999):04d}'
    return f'{state}{district}{series}{number}'


class PlateDataset(Dataset):
    """
    CSV format: image,text
    Example: crops/00001.jpg,MH01CP2655
    Image paths may be absolute or relative to the CSV file's folder.
    """

    def __init__(self, csv_path, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, augment=False):
        self.csv_path = Path(csv_path).resolve()
        self.root = self.csv_path.parent
        self.image_height = image_height
        self.image_width = image_width
        self.rows = []
        with self.csv_path.open('r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {'image', 'text'}.issubset(reader.fieldnames):
                raise ValueError(
                    f'{self.csv_path} must have CSV columns named image,text. '
                    'Example: image,text'
                )
            for row in reader:
                image = (row.get('image') or '').strip()
                text = clean_plate_text(row.get('text') or '')
                if not image or not text:
                    continue
                path = Path(image)
                if not path.is_absolute():
                    resolved = self.root / path
                    if not resolved.is_file():
                        if (self.root / 'crops_backup' / path.name).is_file():
                            resolved = self.root / 'crops_backup' / path.name
                        elif (self.root / 'crops' / path.name).is_file():
                            resolved = self.root / 'crops' / path.name
                    path = resolved
                elif not path.is_file() and (self.root / 'crops_backup' / path.name).is_file():
                    path = self.root / 'crops_backup' / path.name
                self.rows.append((path, text))
        if not self.rows:
            raise ValueError(f'No usable image,text rows found in {self.csv_path}')
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _augment_pil(self, image):
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.3))
        if random.random() < 0.25:
            image = TF.gaussian_blur(image, kernel_size=3)
        if random.random() < 0.35:
            angle = random.uniform(-6, 6)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
        return image

    def __getitem__(self, idx):
        path, text = self.rows[idx]
        if not path.is_file():
            raise FileNotFoundError(f'Image listed in CSV does not exist: {path}')
        image = Image.open(path).convert('RGB')
        image = preprocess_plate_image(image)
        if self.augment:
            image = self._augment_pil(image)
        image = self.normalize(self.to_tensor(image))
        target = torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)
        return image, target, text


class SyntheticPlateDataset(Dataset):
    """On-the-fly printed Indian plates so the CTC model sees clean character shapes."""

    def __init__(self, size=2000, augment=True):
        self.size = size
        self.augment = augment
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        text = random_indian_plate_text()
        image = render_synthetic_plate(text)
        image = preprocess_plate_image(image)
        if self.augment:
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.3:
                image = image.rotate(random.uniform(-4, 4), fillcolor=(128, 128, 128))
        image = self.normalize(self.to_tensor(image))
        target = torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)
        return image, target, text


def ctc_collate(batch):
    images, targets, texts = zip(*batch)
    images = torch.stack(images)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)
    return images, targets, target_lengths, list(texts)
