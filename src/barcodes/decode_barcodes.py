"""
decode_barcodes.py
──────────────────
Decode barcodes from cropped `barcode_region` images using pyzbar (ZBar).
Writes a ground-truth JSON file that maps image stem → decoded string.
This GT is used as the oracle for OCR evaluation.

Usage
-----
    python src/barcodes/decode_barcodes.py \
        --crops   data/crops/barcode_region/ \
        --output  data/barcodes/ground_truth.json \
        --config  config/cfg.yaml

Output JSON structure
---------------------
    {
      "img001": {
        "raw":        "W1234567890",
        "type":       "CODE128",
        "donation_id": "W1234567890",
        "blood_group": null,
        "expiry_date": "31.12.2025"
      },
      ...
    }

Notes on ISBT 128
-----------------
ISBT 128 encodes each data structure in a separate barcode. A blood bag
label typically carries 3–5 barcodes:
  - Donation identification number (DIN)  → starts with flag char + 5-char facility code
  - ABO/Rh blood group                    → 3-char code, e.g. "A0" = A Rh pos
  - Expiry date/time                       → YYYYMMDD[HH]
  - Product code                           → 5 chars

This script decodes all barcodes found in the crop and attempts to
identify each one by its ISBT 128 data structure identifier (first char).
Adjust `ISBT_PARSERS` below if your label format differs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml
from PIL import Image
from pyzbar import pyzbar


# ── ISBT 128 field parsers ────────────────────────────────────────────────────
# Each parser receives the raw decoded string and returns (field_name, value)
# or None if the string does not match.

BLOOD_GROUP_MAP = {
    # ISBT 128 two-char codes → German DIN 58905 notation
    "00": "0 Rh neg",
    "01": "0 Rh neg",     # simplified; full map has ~30 entries
    "02": "0 Rh pos",
    "10": "A Rh neg",
    "11": "A Rh pos",
    "20": "B Rh neg",
    "21": "B Rh pos",
    "30": "AB Rh neg",
    "31": "AB Rh pos",
}

DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})")


def parse_isbt_string(raw: str) -> dict:
    """
    Attempt to extract ISBT 128 fields from a decoded barcode string.
    Returns a dict with field values (None if not found).
    """
    fields = {"donation_id": None, "blood_group": None, "expiry_date": None}

    # Donation identification number: starts with '=' (flag char)
    if raw.startswith("="):
        fields["donation_id"] = raw[1:].strip()
        return fields

    # Expiry date: 8+ digit string YYYYMMDD
    m = DATE_RE.match(raw)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        fields["expiry_date"] = f"{d}.{mo}.{y}"
        return fields

    # Blood group: 2-char code
    if raw[:2] in BLOOD_GROUP_MAP:
        fields["blood_group"] = BLOOD_GROUP_MAP[raw[:2]]
        return fields

    # Unrecognised — store as donation_id by default
    fields["donation_id"] = raw
    return fields


# ── decoding ─────────────────────────────────────────────────────────────────

def decode_image(img_path: Path) -> list[dict]:
    """
    Decode all barcodes found in an image.

    Returns a list of dicts:
        {"raw": str, "type": str, "fields": dict}
    """
    img = Image.open(img_path).convert("RGB")
    decoded = pyzbar.decode(img)

    results = []
    for obj in decoded:
        raw = obj.data.decode("utf-8", errors="replace").strip()
        results.append(
            {
                "raw": raw,
                "type": obj.type,
                "fields": parse_isbt_string(raw),
            }
        )

    return results


def build_ground_truth(crops_dir: Path) -> dict:
    """
    Iterate over all images in crops_dir, decode barcodes, and merge fields.
    Multiple barcodes per image are merged (later barcodes overwrite earlier
    for the same field — adjust logic if your labels carry duplicate fields).
    """
    gt = {}
    image_paths = sorted(
        p for p in crops_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        print(f"  [WARN] No images found in {crops_dir}")
        return gt

    for img_path in image_paths:
        stem = img_path.stem
        decoded_list = decode_image(img_path)

        if not decoded_list:
            print(f"  [WARN] No barcode decoded: {img_path.name}")
            gt[stem] = {"raw": None, "type": None,
                        "donation_id": None, "blood_group": None, "expiry_date": None}
            continue

        # Merge all decoded barcodes for this image
        merged = {"raw": decoded_list[0]["raw"],
                  "type": decoded_list[0]["type"],
                  "donation_id": None, "blood_group": None, "expiry_date": None}
        for d in decoded_list:
            for field, val in d["fields"].items():
                if val is not None:
                    merged[field] = val

        gt[stem] = merged
        print(f"  {img_path.name} → {merged}")

    return gt


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Decode barcodes and write ground truth JSON.")
    parser.add_argument("--crops",  required=True, help="Directory with barcode_region crops")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--config", default="config/cfg.yaml")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Decoding barcodes in {args.crops} …")
    gt = build_ground_truth(Path(args.crops))

    with open(out_path, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    decoded = sum(1 for v in gt.values() if v["raw"] is not None)
    print(f"\nGround truth written → {out_path}")
    print(f"  Total images: {len(gt)}  |  Successfully decoded: {decoded}")


if __name__ == "__main__":
    main()
