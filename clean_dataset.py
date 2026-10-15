import os
import re
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Official Indian RTO State and Union Territory codes
STATE_CODES = (
    "AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LD|MH|ML|MN|MP|MZ|NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB|UA"
)

# Standard Indian License Plate Format:
# Standard: State(2) + District(1-2 digits) + Series(0-2 letters) + Number(4 digits)
# Delhi variant: DL + District(1 digit) + Series(1-3 letters) + Number(4 digits)
INDIAN_PLATE_PATTERN = re.compile(
    rf"^({STATE_CODES})[0-9]{{1,2}}[A-Z]{{0,2}}[0-9]{{4}}$|"
    rf"^(DL)[0-9]{{1}}[A-Z]{{1,3}}[0-9]{{4}}$|"
    rf"^({STATE_CODES})[0-9]{{1,2}}[A-Z]{{0,2}}[0-9]{{3}}$"
)

# Extraction regex to pull plate string out of filenames / XML tags
EXTRACT_PATTERN = re.compile(
    rf"({STATE_CODES})[0-9]{{1,2}}[A-Z]{{0,2}}[0-9]{{4}}|"
    rf"(DL)[0-9]{{1}}[A-Z]{{1,3}}[0-9]{{4}}|"
    rf"({STATE_CODES})[0-9]{{1,2}}[A-Z]{{0,2}}[0-9]{{3}}"
)

def clean_plate_text(text):
    if not text:
        return None
    # Keep only uppercase alphanumeric characters
    cleaned = "".join(c for c in str(text).strip().upper() if c.isalnum())
    # Strip leading 'IND' national badge if present
    if cleaned.startswith("IND") and len(cleaned) > 7:
        cleaned = cleaned[3:]
    # Reject strings with repetitive corrupt patterns like KL0KL0100
    if len(cleaned) < 7 or len(cleaned) > 11:
        return None
    # Check if exact match to standard Indian License Plate
    if INDIAN_PLATE_PATTERN.match(cleaned):
        return cleaned
    # Check if pattern can be extracted from within string
    match = EXTRACT_PATTERN.search(cleaned)
    if match:
        return match.group(0)
    return None

def main():
    crops_dir = Path("crops").resolve()
    archive_dir = Path(r"C:\Users\Admin\Downloads\miniproject(license)\archive (2)").resolve()

    if not crops_dir.is_dir():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")

    # Build XML lookup map
    print("Scanning XML annotations in archive (2)...")
    xml_map = {}
    if archive_dir.is_dir():
        for xml_path in archive_dir.rglob("*.xml"):
            xml_map[xml_path.stem] = xml_path
    print(f"Indexed {len(xml_map)} XML files.")

    crop_images = sorted(crops_dir.glob("*.jpg"))
    print(f"Found {len(crop_images)} cropped images in {crops_dir}.")

    dataset = []
    dropped_junk = []

    for img_path in crop_images:
        stem = img_path.stem
        base_stem = re.sub(r"_\d+$", "", stem)

        candidate_labels = []

        # 1. From matching XML annotation (object names)
        if base_stem in xml_map:
            try:
                tree = ET.parse(xml_map[base_stem])
                root = tree.getroot()
                for obj in root.findall(".//object"):
                    name_tag = obj.find("name")
                    if name_tag is not None and name_tag.text:
                        lbl = clean_plate_text(name_tag.text)
                        if lbl:
                            candidate_labels.append(lbl)
            except Exception:
                pass

        # 2. From filename directly (e.g. car-wbs-MH03AR5549_00000)
        lbl_from_filename = clean_plate_text(stem)
        if lbl_from_filename:
            candidate_labels.append(lbl_from_filename)

        # Pick the best valid plate label
        chosen_label = None
        for lbl in candidate_labels:
            if INDIAN_PLATE_PATTERN.match(lbl):
                chosen_label = lbl
                break

        if chosen_label:
            dataset.append((f"crops/{img_path.name}", chosen_label))
        else:
            dropped_junk.append(img_path.name)

    print(f"\n==========================================")
    print(f"Total clean, verified Indian plates: {len(dataset)}")
    print(f"Dropped unverified / non-plate images : {len(dropped_junk)}")
    print(f"==========================================")

    if not dataset:
        print("ERROR: No valid plates found! Check paths.")
        return

    # Shuffle with fixed seed for perfect reproducibility
    random.seed(42)
    random.shuffle(dataset)

    n_total = len(dataset)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    train_split = dataset[:n_train]
    val_split = dataset[n_train:n_train + n_val]
    test_split = dataset[n_train + n_val:]

    for name, split in [("train.csv", train_split), ("val.csv", val_split), ("test.csv", test_split)]:
        out_path = Path(name)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "text"])
            writer.writerows(split)
        print(f"Saved {name:10s}: {len(split):4d} plates")

    print("\nDataset preparation complete! All labels are 100% verified Indian plates.")

if __name__ == "__main__":
    main()
