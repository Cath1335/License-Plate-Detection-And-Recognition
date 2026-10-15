from pathlib import Path
import sys
import cv2
import torch
import easyocr
import re
from collections import Counter

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "train" / "lse_pretrained_30epochs" / "weights" / "best.pt"

if len(sys.argv) < 2:
    print("Usage: python plate_ocr.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.exists():
    print("Image not found:", image_path)
    sys.exit(1)

print("Loading license plate detector...")

model = torch.hub.load(
    str(ROOT),
    "custom",
    path=str(WEIGHTS),
    source="local"
)

model.conf = 0.25

print("Loading OCR...")
reader = easyocr.Reader(["en"], gpu=False)

image = cv2.imread(str(image_path))

if image is None:
    print("Could not read image.")
    sys.exit(1)


def make_versions(crop):

    versions = []

    # Original enlarged 4x
    up4 = cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    versions.append(("original_4x", up4))

    # Original enlarged 6x
    up6 = cv2.resize(
        crop,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    versions.append(("original_6x", up6))

    # Grayscale
    gray = cv2.cvtColor(up6, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(gray)

    versions.append(("clahe", clahe_img))

    # Sharpen
    bilateral = cv2.bilateralFilter(
        clahe_img,
        9,
        75,
        75
    )

    blur = cv2.GaussianBlur(
        bilateral,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        bilateral,
        1.7,
        blur,
        -0.7,
        0
    )

    versions.append(("sharpened", sharpened))

    # Otsu
    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    versions.append(("otsu", otsu))

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    versions.append(("adaptive", adaptive))

    return versions


def clean_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


def similarity(a, b):

    if not a or not b:
        return 0

    length = max(len(a), len(b))

    matches = 0

    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            matches += 1

    return matches / length


def choose_consensus(candidates):
    if not candidates:
        return "", 0.0, ""

    # Actual candidate format:
    # (confidence, text, method_name)

    counts = Counter(text for confidence, text, method in candidates)

    scored = []

    for confidence, text, method in candidates:
        frequency = counts[text]

        # Frequency has higher priority than individual OCR confidence
        score = (frequency * 1.0) + (confidence * 0.3)

        scored.append(
            (score, frequency, confidence, method, text)
        )

    # First: frequency
    # Second: confidence
    scored.sort(
        key=lambda x: (x[1], x[2]),
        reverse=True
    )

    best_score, best_frequency, best_confidence, best_method, best_text = scored[0]

    print("\nOCR consensus:")
    for text, count in counts.most_common():
        print(f"  {text}: {count} occurrence(s)")

    return best_text, best_confidence, best_method

results = model(image)

detections = results.xyxy[0].cpu().numpy()

output = image.copy()

found = False

# Select only the highest-confidence detection
if len(detections) > 0:

    best_detection = max(
        detections,
        key=lambda det: det[4]
    )

    x1, y1, x2, y2, detection_conf, class_id = best_detection

    x1, y1, x2, y2 = map(
        int,
        [x1, y1, x2, y2]
    )

    detection_conf = float(detection_conf)

    print(
        f"Best plate detection: "
        f"confidence={detection_conf:.3f}"
    )

    # Crop the best detected plate
    plate_crop = image[y1:y2, x1:x2]

    found = True


# Use only the best-confidence detection
if found:
        # Add padding around detected plate
    pad = 10

    crop = image[
        max(0, y1 - pad):min(image.shape[0], y2 + pad),
        max(0, x1 - pad):min(image.shape[1], x2 + pad)
    ]

    if crop.size != 0:

        print()
        print("Detected plate crop size:", crop.shape)

    # Upscale small plate crops before OCR
        h, w = crop.shape[:2]

        if h < 50:
            scale = 6
        else:
            scale = 4

        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        print("Upscaled crop size:", crop.shape)

        versions = make_versions(crop)

        candidates = []

    for method_name, processed in versions:

        ocr_results = reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        for result in ocr_results:

            text = clean_text(result[1])
            confidence = float(result[2])

            if not text:
                continue

            candidates.append(
                (
                    confidence,
                    text,
                    method_name
                )
            )

    # Process OCR results only after all preprocessing versions
    if candidates:

        print()
        print("OCR candidates:")

        sorted_candidates = sorted(
            candidates,
            key=lambda x: x[0],
            reverse=True
        )

        for confidence, text, method_name in sorted_candidates:

            print(
                f"  {method_name}: "
                f"{text} "
                f"(confidence {confidence:.3f})"
            )

        selected = choose_consensus(candidates)

        if selected:

            best_text, best_conf, best_method = selected

            print()
            print("Selected OCR result:", best_text)
            print(f"OCR confidence: {best_conf:.3f}")
            print(f"Detection confidence: {detection_conf:.3f}")
            print(f"OCR method: {best_method}")

            plate_text = best_text

        else:

            plate_text = "Plate detected"

    else:

        plate_text = "Plate detected"

        print("OCR could not recognize text.")

    # Draw detection box
    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw OCR text
    cv2.putText(
        output,
        plate_text,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


if not found:

    print("No license plate detected.")


output_dir = ROOT / "runs" / "detect"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = output_dir / "ocr_result.jpg"

cv2.imwrite(
    str(output_path),
    output
)

print()
print("Output image saved to:")
print(output_path)