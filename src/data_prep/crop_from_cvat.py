"""
crop_from_cvat.py
─────────────────
Parse a CVAT XML annotation file (task-level export, not project-level),
crop every labelled bounding box from the corresponding image, and save
crops organised by label name.

Usage
-----
    python src/data_prep/crop_from_cvat.py \
        --annotations data/cvat_export/annotations.xml \
        --images      data/raw/ \
        --output      data/crops/ \
        --config      config/cfg.yaml

Output layout
-------------
    data/crops/
    ├── donation_id/
    │   ├── img001_donation_id_0.png
    │   └── ...
    ├── blood_group/
    │   └── ...
    └── expiry_date/
        └── ...

A companion JSON index is written to data/crops/index.json with entries:
    {
      "image": "img001.jpg",
      "label": "donation_id",
      "crop_path": "data/crops/donation_id/img001_donation_id_0.png",
      "box_xyxy": [x1, y1, x2, y2]
    }
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from lxml import etree
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def parse_cvat_xml(xml_path: str) -> list[dict]:
    """
    Return a flat list of annotation records from a CVAT XML file.

    Each record is:
        {
            "image_name": str,
            "label":      str,
            "box_xyxy":   (x1, y1, x2, y2),   # integers
        }
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    records = []

    for image_el in root.iter("image"):
        img_name = image_el.get("name")
        for box_el in image_el.iter("box"):
            label = box_el.get("label")
            x1 = int(float(box_el.get("xtl")))
            y1 = int(float(box_el.get("ytl")))
            x2 = int(float(box_el.get("xbr")))
            y2 = int(float(box_el.get("ybr")))
            records.append(
                {
                    "image_name": img_name,
                    "label": label,
                    "box_xyxy": (x1, y1, x2, y2),
                }
            )

    return records


def find_image(image_dir: Path, name: str) -> Path | None:
    """Locate an image file by name (searches common extensions)."""
    for ext in ("", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        candidate = image_dir / (name + ext)
        if candidate.exists():
            return candidate
        # handle case where 'name' already carries an extension
        candidate2 = image_dir / name
        if candidate2.exists():
            return candidate2
    return None


def crop_and_save(
    records: list[dict],
    image_dir: Path,
    output_dir: Path,
    active_labels: set[str],
    padding: int = 2,
) -> list[dict]:
    """
    Crop bounding boxes from their source images and save as PNG.

    Parameters
    ----------
    records       : parsed annotation records
    image_dir     : directory containing raw images
    output_dir    : root output directory (subdirs created per label)
    active_labels : only process these label names
    padding       : pixel padding added around each box (clamped to image edges)

    Returns
    -------
    index : list of dicts suitable for JSON serialisation
    """
    index = []
    label_counters: dict[str, int] = {}

    for rec in records:
        label = rec["label"]
        if label not in active_labels:
            continue

        img_path = find_image(image_dir, rec["image_name"])
        if img_path is None:
            print(f"  [WARN] image not found: {rec['image_name']}")
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = rec["box_xyxy"]

        # apply padding (clamped)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop = img.crop((x1, y1, x2, y2))

        # build output path
        stem = Path(rec["image_name"]).stem
        count = label_counters.get(label, 0)
        label_counters[label] = count + 1
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        crop_name = f"{stem}_{label}_{count}.png"
        crop_path = label_dir / crop_name
        crop.save(crop_path)

        index.append(
            {
                "image": str(img_path.relative_to(image_dir.parent)),
                "label": label,
                "crop_path": str(crop_path),
                "box_xyxy": list(rec["box_xyxy"]),
            }
        )

    return index


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Crop CVAT-annotated regions.")
    parser.add_argument("--annotations", required=True, help="Path to CVAT annotations.xml")
    parser.add_argument("--images",      required=True, help="Directory with raw images")
    parser.add_argument("--output",      required=True, help="Root output directory for crops")
    parser.add_argument("--config",      default="config/cfg.yaml")
    parser.add_argument("--padding",     type=int, default=2, help="Pixel padding per box")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label_settings = cfg["annotation"]["label_settings"]

    # labels that should be cropped (include: true, whether OCR is needed or not)
    active_labels = {
        label
        for label, settings in label_settings.items()
        if settings.get("include", True)
    }

    print(f"Parsing {args.annotations} …")
    records = parse_cvat_xml(args.annotations)
    print(f"  Found {len(records)} annotated boxes.")

    print(f"Cropping to {args.output} …")
    index = crop_and_save(
        records=records,
        image_dir=Path(args.images),
        output_dir=Path(args.output),
        active_labels=active_labels,
        padding=args.padding,
    )

    # write index
    index_path = Path(args.output) / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. {len(index)} crops saved. Index → {index_path}")

    # quick summary
    from collections import Counter
    counts = Counter(r["label"] for r in index)
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n} crops")


if __name__ == "__main__":
    main()
