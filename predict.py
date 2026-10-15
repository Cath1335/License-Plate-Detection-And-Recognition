import warnings
warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from dataset import preprocess_plate_image
from model import CNNGRUCTC, IMAGE_HEIGHT, IMAGE_WIDTH, greedy_decode


def load_image_path(image_path):
    image_path = Path(image_path)
    candidates = [
        image_path,
        image_path.with_suffix('.jpg'),
        image_path.with_suffix('.png'),
        Path('crops') / image_path.name,
        Path('crops') / f'{image_path.name}.jpg',
        Path('crops') / f'{image_path.name}.png',
        Path('../detection_crop') / image_path.name,
        Path('../detection_crop') / f'{image_path.name}.jpg',
        Path('../detection_crop') / f'{image_path.name}.png',
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return image_path


def main():
    parser = argparse.ArgumentParser(description='Recognize text in a cropped license plate.')
    parser.add_argument('--image', required=True, help='Path to cropped plate image')
    parser.add_argument('--weights', default='checkpoints/best.pth')
    parser.add_argument('--out', default='recognition_output.txt')
    args = parser.parse_args()

    image_path = load_image_path(args.image)
    weights_path = Path(args.weights)
    if not image_path.is_file():
        print(f'\n[Error] Image not found: {image_path}\n')
        return
    if not weights_path.is_file():
        raise FileNotFoundError(
            f'Checkpoint not found: {weights_path}. Train first with train.py.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = CNNGRUCTC().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    height = int(checkpoint.get('image_height', IMAGE_HEIGHT))
    width = int(checkpoint.get('image_width', IMAGE_WIDTH))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    image = preprocess_plate_image(Image.open(image_path).convert('RGB'))
    if image.size != (width, height):
        image = image.resize((width, height))
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        conf = logits.softmax(dim=2).max(dim=2).values.mean().item() * 100.0
        prediction = greedy_decode(logits.cpu())[0]

    print('-' * 45)
    print(f'Image                : {image_path.name}')
    print(f'Predicted plate text : {prediction}')
    print(f'Confidence           : {conf:.1f}%')
    print('-' * 45)
    Path(args.out).write_text(
        f'Image: {image_path}\nPredicted plate text: {prediction}\nConfidence: {conf:.1f}%\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
