"""
barcode_vs_ocr.py
─────────────────
End-to-end oracle validation script for the feasibility test.

What it does
------------
1. Reads the CVAT XML annotation file.
2. For every annotated image:
   a. Crops each `barcode` box and decodes it with pyzbar (ZBar).
   b. Parses the raw barcode string into ISBT 128 fields:
        product_id, blood_group, expiry_date
   c. Crops each text field box (product_id / blood_group / expiry_date).
   d. Runs OCR on each text crop (Tesseract by default; EasyOCR optional).
   e. Compares barcode-decoded value vs OCR-read value for each field.
3. Writes:
   - results/barcode_vs_ocr.json  — full per-image detail
   - results/barcode_vs_ocr.csv   — one row per (image, field) for analysis
4. Prints a summary table to stdout.

This is the go/no-go oracle check: if barcode decoding succeeds on ≥ 80 % of
images AND the barcode field values match the printed text (CER < 0.15), the
barcode-as-oracle pipeline is trustworthy and the project can proceed.

Usage
-----
    # Tesseract only (fastest, no extra deps):
    python src/barcodes/barcode_vs_ocr.py \\
        --annotations data/cvat_export/annotations.xml \\
        --images      data/raw/ \\
        --engine      tesseract

    # Add EasyOCR as a second engine for comparison:
    python src/barcodes/barcode_vs_ocr.py \\
        --annotations data/cvat_export/annotations.xml \\
        --images      data/raw/ \\
        --engine      tesseract easyocr

    # Write results to a custom directory:
    python src/barcodes/barcode_vs_ocr.py \\
        --annotations data/cvat_export/annotations.xml \\
        --images      data/raw/ \\
        --output      results/feasibility/

Dependencies
------------
    pip install pyzbar pytesseract easyocr editdistance lxml Pillow numpy
    sudo apt install libzbar0 tesseract-ocr tesseract-ocr-deu
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

import editdistance
import numpy as np
from lxml import etree
from PIL import Image, ImageEnhance, ImageFilter

# ── ISBT 128 blood-group map (subset; extend if your labels use other codes) ──

BLOOD_GROUP_MAP: dict[str, str] = {
    "00": "0 Rh neg",
    "01": "0 Rh neg",
    "02": "0 Rh pos",
    "03": "0 Rh pos",
    "10": "A Rh neg",
    "11": "A Rh pos",
    "12": "A Rh neg",
    "13": "A Rh pos",
    "20": "B Rh neg",
    "21": "B Rh pos",
    "22": "B Rh neg",
    "23": "B Rh pos",
    "30": "AB Rh neg",
    "31": "AB Rh pos",
    "32": "AB Rh neg",
    "33": "AB Rh pos",
}

# Regex for YYYYMMDD dates inside barcode payloads
_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")

# Labels that carry text we want to compare against
TEXT_LABELS = ("product_id", "blood_group", "expiry_date")
BARCODE_LABEL = "barcode"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — parse CVAT XML into a structured dict
# ─────────────────────────────────────────────────────────────────────────────

def parse_cvat_xml(xml_path: Path) -> dict[str, dict]:
    """
    Parse a CVAT-for-images XML export.

    Returns
    -------
    {
      "img001.jpg": {
        "barcode": [(x1,y1,x2,y2), ...],   # may be several barcodes
        "product_id":    [(x1,y1,x2,y2), ...],
        "blood_group":    [(x1,y1,x2,y2), ...],
        "expiry_date":    [(x1,y1,x2,y2), ...],
      },
      ...
    }
    """
    tree = etree.parse(str(xml_path))
    images: dict[str, dict] = {}
    

    for img_el in tree.getroot().iter("image"):
        name = img_el.get("name")
        # entry: dict[str, list] = {
        #     BARCODE_LABEL: [],
        #     "product_id": [],
        #     "blood_group": [],
        #     "expiry_date": [],
        # }
        entry: dict[str, list] = {}
        # Store XML-recorded dimensions for EXIF rotation diagnostics
        try:
            entry["_xml_width"]  = int(img_el.get("width",  0))
            entry["_xml_height"] = int(img_el.get("height", 0))
        except (TypeError, ValueError):
            pass

        for box in img_el.iter("box"):
            label = box.get("label")
            # if label not in entry:
            #     continue
            if label is None:
                continue
            if label not in entry:
                entry[label] = []
            x1 = int(float(box.get("xtl")))
            y1 = int(float(box.get("ytl")))
            x2 = int(float(box.get("xbr")))
            y2 = int(float(box.get("ybr")))
            entry[label].append((x1, y1, x2, y2))
        images[name] = entry

    return images


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — barcode decoding
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_for_barcode(img: Image.Image) -> list[Image.Image]:
    """
    Return several preprocessing variants of an image to maximise pyzbar
    decode success rate on mobile-camera shots.
    """
    variants = [img]  # original first

    # Grayscale + contrast boost
    gray = img.convert("L")
    gray_contrast = ImageEnhance.Contrast(gray).enhance(2.0)
    variants.append(gray_contrast)

    # Upscale 2× (helps with small/distant barcodes)
    w, h = img.size
    upscaled = gray_contrast.resize((w * 2, h * 2), Image.LANCZOS)
    variants.append(upscaled)

    # Slight sharpening
    sharpened = gray_contrast.filter(ImageFilter.SHARPEN)
    variants.append(sharpened)

    return variants


def decode_barcode_crop(crop: Image.Image) -> list[dict]:
    """
    Try to decode barcodes from a single crop image.
    Returns a list of decoded barcode dicts (may be empty).
    """
    from pyzbar import pyzbar

    for variant in _preprocess_for_barcode(crop):
        decoded = pyzbar.decode(variant)
        if decoded:
            return [
                {
                    "raw":  obj.data.decode("utf-8", errors="replace").strip(),
                    "type": obj.type,
                }
                for obj in decoded
            ]
    return []


def parse_isbt128(raw: str) -> dict[str, Optional[str]]:
    """
    Parse a single raw barcode string into ISBT 128 field values.

    Returns dict with keys: product_id, blood_group, expiry_date.
    Unrecognised strings are stored under product_id as a fallback.
    """
    fields: dict[str, Optional[str]] = {
        "product_id": None,
        "blood_group": None,
        "expiry_date": None,
    }

    raw_stripped = raw.strip()

    # ── Product identification number ────────────────────────────────────────
    # Format: !2768022 + 11 digits (18 chars total)
    # The prefix !276 is the German DRK facility identifier block.
    # Reject !P... (product codes) and short strings.
    if re.match(r"^!276\d{13}$", raw_stripped):
        fields["product_id"] = raw_stripped
        return fields

    # if re.match(r"^[A-Z][0-9]{10}$", raw_stripped):
    #     fields["product_id"] = raw_stripped
    #     return fields

    # ── Expiry / collection date ───────────────────────────────────────────────
    # !E + YYYYMMDD format
    if raw_stripped.startswith("!E"):
        date_str = raw_stripped[2:10]  # Extract YYYYMMDD after !E
        m = re.match(r"(\d{4})(\d{2})(\d{2})", date_str)
        if m:
            y, mo, d = m.groups()
            try:
                from datetime import date
                date(int(y), int(mo), int(d))  # Validate
                fields["expiry_date"] = f"{d}.{mo}.{y}"  # → DD.MM.YYYY
            except ValueError:
                pass
            return fields

    # ── Blood group ───────────────────────────────────────────────────────────
    # ISBT 128 codes with !R prefix 
    if raw_stripped.startswith("!R"):
        #code = raw_stripped[2:6]  # Extract 4-char code after !R
        # blood_map = {
        #     "1132": "A",
        #     "2132": "B", 
        #     "3152": "AB",
        #     "4132": "O",
        #     # Add O+: "0132", O-: "0133", etc. as needed
        # }
        #fields["blood_group"] = blood_map.get(code)
        fields["blood_group"] = "!R" + raw_stripped 
        return fields

    # ── Fallback: store as product_id ────────────────────────────────────────
    #fields["product_id"] = raw_stripped
    return fields


def decode_all_barcodes(
    img: Image.Image,
    barcode_boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    padding: int = 4,
) -> dict[str, Optional[str]]:
    """
    Crop every barcode_region box, decode with pyzbar, and merge all
    ISBT 128 fields found across all barcodes on one label.

    Returns merged dict: {product_id, blood_group, expiry_date, _raw_list}
    """
    w, h = image_size
    # Collect ALL decoded values across all barcode boxes first,
    # then pick the best candidate per field (most specific wins).

    all_parsed: list[dict] = []
    raw_list: list[dict] = []

    for box in barcode_boxes:
        x1, y1, x2, y2 = box
        
        # apply padding, clamp to image
        x1c = max(0, x1 - padding)
        y1c = max(0, y1 - padding)
        x2c = min(w, x2 + padding)
        y2c = min(h, y2 + padding)
        # Safety check — print diagnostic if crop is still invalid after clamping
        if x1c >= x2c or y1c >= y2c:
            print(f"    [WARN] Invalid crop skipped: box=({x1},{y1},{x2},{y2}) "
                  f"→ clamped=({x1c},{y1c},{x2c},{y2c}) "
                  f"image_size=({w},{h})")
            continue
        crop = img.crop((x1c, y1c, x2c, y2c))
        
        decoded = decode_barcode_crop(crop)

        for d in decoded:
            raw_list.append(d)
            all_parsed.append(parse_isbt128(d["raw"]))

    # Merge: for each field, take the first non-None value found.
    # parse_isbt128 now returns None for non-matching barcodes,
    # so the merge naturally picks the correct barcode type.
    merged: dict[str, Optional[str]] = {
        "product_id": None,
        "blood_group": None,
        "expiry_date": None,
    }
    for parsed in all_parsed:
        for field, val in parsed.items():
            if val is not None and merged[field] is None:
                merged[field] = val

    merged["_raw_list"] = raw_list
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — OCR on text field crops
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Light preprocessing that consistently helps OCR on label crops:
    - Convert to grayscale
    - Boost contrast
    - Upscale to at least 48 px tall (Tesseract optimal height)
    """
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    w, h = gray.size
    if h < 48:
        scale = 48 / h
        gray = gray.resize((int(w * scale), 48), Image.LANCZOS)
    return gray


def ocr_tesseract(img: Image.Image) -> tuple[str, float]:
    """Returns (predicted_text, latency_ms)."""
    import pytesseract
    processed = _preprocess_for_ocr(img)
    t0 = time.perf_counter()
    text = pytesseract.image_to_string(
        processed,
        lang="deu",
        config="--psm 7 --oem 1",
    ).strip()
    ms = (time.perf_counter() - t0) * 1000
    return text, ms


_easyocr_reader = None


def ocr_easyocr(img: Image.Image) -> tuple[str, float]:
    """Returns (predicted_text, latency_ms). Lazy-loads the reader on first call."""
    global _easyocr_reader
    import easyocr
    if _easyocr_reader is None:
        print("    [EasyOCR] loading model (one-time download ~50 MB) …")
        _easyocr_reader = easyocr.Reader(["de", "en"], gpu=False, verbose=False)

    processed = _preprocess_for_ocr(img).convert("RGB")
    arr = np.array(processed)
    t0 = time.perf_counter()
    result = _easyocr_reader.readtext(arr, detail=0)
    ms = (time.perf_counter() - t0) * 1000
    return " ".join(result).strip(), ms


OCR_ENGINES = {
    "tesseract": ocr_tesseract,
    "easyocr":   ocr_easyocr,
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — comparison metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_cer(hyp: str, ref: str) -> float:
    """Normalised Levenshtein distance at character level [14]."""
    hyp = hyp.strip().lower()
    ref = ref.strip().lower()
    if not ref:
        return 0.0 if not hyp else 1.0
    return editdistance.eval(hyp, ref) / len(ref)


def match_status(cer_score: float) -> str:
    """Human-readable match label for the summary table."""
    if cer_score == 0.0:
        return "EXACT"
    if cer_score <= 0.10:
        return "NEAR"
    if cer_score <= 0.30:
        return "PARTIAL"
    return "MISMATCH"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — per-image pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    img_name: str,
    annotation: dict,
    image_dir: Path,
    engines: list[str],
) -> dict:
    """
    Run the full barcode → OCR → comparison pipeline for one image.

    Returns a result dict ready for JSON serialisation.
    """
    result: dict = {
        "image": img_name,
        "barcode": {"decoded": False, "raw_list": [], "fields": {}},
        "ocr": {},
        "comparison": {},
    }

    # ── Load image ────────────────────────────────────────────────────────────
    # Strategy:
    # 1. Try the exact filename from the XML (handles the common case).
    # 2. If that fails, strip the extension and try all common image extensions.
    #    This fixes mismatches where the XML says .jpeg but the file is .jpg
    #    or where CVAT strips/changes the extension on export.
    img_path = None
    stem = Path(img_name).stem   # e.g. "20251118_100349"  (no extension)
 
    candidates = [
        image_dir / img_name,                    # exact name from XML
        image_dir / (stem + ".jpg"),
        image_dir / (stem + ".jpeg"),
        image_dir / (stem + ".JPG"),
        image_dir / (stem + ".JPEG"),
        image_dir / (stem + ".png"),
        image_dir / (stem + ".PNG"),
        image_dir / (stem + ".tiff"),
        image_dir / (stem + ".tif"),
    ]
 
    for candidate in candidates:
        if candidate.exists():
            img_path = candidate
            break
 
    if img_path is None:
        result["error"] = f"Image not found: {img_name}"
        print(f"  [SKIP] {img_name} — not found in {image_dir}")
        print(f"         Tried: {[str(c.name) for c in candidates]}")
        return result

    # img = Image.open(img_path).convert("RGB")
    # w, h = img.size
    img = Image.open(img_path)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    w, h = img.size
    xml_w = annotation.get("_xml_width")
    xml_h = annotation.get("_xml_height")
    if xml_w and xml_h and (w != xml_w or h != xml_h):
        print(f"  [WARN] {img_name}: XML {xml_w}x{xml_h} vs PIL {w}x{h} — EXIF rotation detected.")


    # ── Decode barcodes ───────────────────────────────────────────────────────
    barcode_boxes = annotation.get(BARCODE_LABEL, [])
    if not barcode_boxes:
        result["barcode"]["error"] = "no barcode_region annotations"
        print(f"  [WARN] {img_name} — no barcode_region boxes annotated")
    else:
        bc = decode_all_barcodes(img, barcode_boxes, (w, h))
        raw_list = bc.pop("_raw_list", [])
        result["barcode"]["decoded"] = bool(raw_list)
        result["barcode"]["raw_list"] = raw_list
        result["barcode"]["fields"] = bc
        if raw_list:
            print(f"  {img_name} — barcode: {[d['raw'] for d in raw_list]}")
        else:
            print(f"  [WARN] {img_name} — pyzbar found no barcode in {len(barcode_boxes)} box(es)")

    barcode_fields = result["barcode"]["fields"]

    # ── OCR each text field ───────────────────────────────────────────────────
    for engine_name in engines:
        engine_fn = OCR_ENGINES.get(engine_name)
        if engine_fn is None:
            print(f"  [WARN] Unknown engine '{engine_name}' — skipping")
            continue

        result["ocr"][engine_name] = {}

        for field in TEXT_LABELS:
            boxes = annotation.get(field, [])
            if not boxes:
                result["ocr"][engine_name][field] = {
                    "text": None, "latency_ms": None,
                    "note": "no annotation box",
                }
                continue

            # If multiple boxes, take the first (should be one per field)
            x1, y1, x2, y2 = boxes[0]
            
            x1c = max(0, x1 - 2)
            y1c = max(0, y1 - 2)
            x2c = min(w, x2 + 2)
            y2c = min(h, y2 + 2)
            crop = img.crop((x1c, y1c, x2c, y2c))
            
            try:
                ocr_text, latency_ms = engine_fn(crop)
            except Exception as exc:
                ocr_text, latency_ms = "", 0.0
                print(f"    [ERROR] {engine_name}/{field}: {exc}")

            result["ocr"][engine_name][field] = {
                "text":       ocr_text,
                "latency_ms": round(latency_ms, 1),
            }

    # ── Compare barcode fields vs OCR fields ──────────────────────────────────
    for engine_name in engines:
        if engine_name not in result["ocr"]:
            continue
        result["comparison"][engine_name] = {}

        for field in TEXT_LABELS:
            bc_val  = barcode_fields.get(field)
            ocr_val = result["ocr"][engine_name].get(field, {}).get("text")

            if bc_val is None:
                status = {
                    "barcode_value": None,
                    "ocr_value":     ocr_val,
                    "cer":           None,
                    "match":         "NO_BARCODE_GT",
                }
            elif ocr_val is None:
                status = {
                    "barcode_value": bc_val,
                    "ocr_value":     None,
                    "cer":           None,
                    "match":         "NO_OCR",
                }
            else:
                c = compute_cer(ocr_val, bc_val)
                status = {
                    "barcode_value": bc_val,
                    "ocr_value":     ocr_val,
                    "cer":           round(c, 4),
                    "match":         match_status(c),
                }

            result["comparison"][engine_name][field] = status

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — aggregate and report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], engines: list[str]) -> None:
    """Print a structured summary table to stdout."""
    total = len(all_results)
    decoded = sum(1 for r in all_results if r["barcode"].get("decoded"))

    print("\n" + "═" * 72)
    print(f"  BARCODE DECODE RATE: {decoded}/{total} images "
          f"({100*decoded/total:.0f}%)")
    print("═" * 72)

    for engine in engines:
        print(f"\n  Engine: {engine}")
        print(f"  {'Field':<18} {'N':>4} {'Mean CER':>10} {'EXACT':>7} "
              f"{'NEAR':>7} {'MISMATCH':>9} {'p95 ms':>8}")
        print("  " + "─" * 68)

        for field in TEXT_LABELS:
            cer_scores, latencies = [], []
            counts = {"EXACT": 0, "NEAR": 0, "PARTIAL": 0, "MISMATCH": 0}

            for r in all_results:
                cmp = r.get("comparison", {}).get(engine, {}).get(field, {})
                ocr = r.get("ocr", {}).get(engine, {}).get(field, {})

                if cmp.get("cer") is not None:
                    cer_scores.append(cmp["cer"])
                    m = cmp.get("match", "MISMATCH")
                    counts[m] = counts.get(m, 0) + 1

                lat = ocr.get("latency_ms")
                if lat is not None:
                    latencies.append(lat)

            n = len(cer_scores)
            mean_cer = f"{sum(cer_scores)/n:.3f}" if n else "—"
            p95 = f"{sorted(latencies)[int(len(latencies)*0.95)]:.0f}" \
                  if latencies else "—"

            print(f"  {field:<18} {n:>4} {mean_cer:>10} "
                  f"{counts['EXACT']:>7} {counts['NEAR']:>7} "
                  f"{counts['MISMATCH']:>9} {p95:>8}")

    # Go / no-go verdict
    print("\n" + "═" * 72)
    print("  GO / NO-GO SIGNALS")
    print("─" * 72)
    decode_rate = decoded / total if total else 0
    print(f"  Barcode decode rate: {decode_rate:.0%}  "
          f"{'✓ GO' if decode_rate >= 0.8 else '✗ INVESTIGATE — fix crop boxes or image quality'}")

    for engine in engines:
        all_cer = []
        for r in all_results:
            for field in TEXT_LABELS:
                c = r.get("comparison", {}).get(engine, {}).get(field, {}).get("cer")
                if c is not None:
                    all_cer.append(c)
        if all_cer:
            overall = sum(all_cer) / len(all_cer)
            verdict = "✓ GO" if overall < 0.15 else \
                      "~ MARGINAL — consider fine-tuning" if overall < 0.30 else \
                      "✗ NO-GO — engine unsuitable zero-shot"
            print(f"  [{engine}] mean CER (all fields): {overall:.3f}  {verdict}")

    all_lat = []
    for r in all_results:
        for engine in engines:
            for field in TEXT_LABELS:
                lat = r.get("ocr", {}).get(engine, {}).get(field, {}).get("latency_ms")
                if lat:
                    all_lat.append(lat)
    if all_lat:
        p95_all = sorted(all_lat)[int(len(all_lat) * 0.95)]
        print(f"  p95 latency (this machine): {p95_all:.0f} ms  "
              f"{'✓ headroom for ARM' if p95_all < 200 else '~ expect 3–5× slower on device'}")
    print("═" * 72)


def write_csv(all_results: list[dict], engines: list[str], out_path: Path) -> None:
    """Write one row per (image, engine, field) to a CSV file."""
    rows = []
    for r in all_results:
        img = r["image"]
        bc_decoded = r["barcode"].get("decoded", False)
        bc_raw = ", ".join(d["raw"] for d in r["barcode"].get("raw_list", []))

        for engine in engines:
            for field in TEXT_LABELS:
                ocr_info = r.get("ocr", {}).get(engine, {}).get(field, {})
                cmp_info = r.get("comparison", {}).get(engine, {}).get(field, {})
                rows.append({
                    "image":          img,
                    "engine":         engine,
                    "field":          field,
                    "barcode_decoded":bc_decoded,
                    "barcode_raw":    bc_raw,
                    "barcode_value":  cmp_info.get("barcode_value"),
                    "ocr_text":       ocr_info.get("text"),
                    "cer":            cmp_info.get("cer"),
                    "match":          cmp_info.get("match"),
                    "latency_ms":     ocr_info.get("latency_ms"),
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop barcodes, decode with pyzbar, compare to OCR text fields.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/barcodes/barcode_vs_ocr.py \\
      --annotations data/cvat_export/annotations.xml \\
      --images data/raw/

  python src/barcodes/barcode_vs_ocr.py \\
      --annotations data/cvat_export/annotations.xml \\
      --images data/raw/ \\
      --engine tesseract easyocr \\
      --output results/feasibility/
        """,
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Path to CVAT annotations.xml",
    )
    parser.add_argument(
        "--images", required=True,
        help="Directory containing raw images",
    )
    parser.add_argument(
        "--engine", nargs="+", default=["tesseract"],
        choices=list(OCR_ENGINES.keys()),
        help="OCR engine(s) to use (default: tesseract)",
    )
    parser.add_argument(
        "--output", default="results/",
        help="Output directory for JSON and CSV (default: results/)",
    )
    parser.add_argument(
        "--save_crops", action="store_true",
        help="Save barcode and field crops to output/crops/ for inspection",
    )
    args = parser.parse_args()

    xml_path   = Path(args.annotations)
    image_dir  = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse annotations ─────────────────────────────────────────────────────
    print(f"\nParsing {xml_path} …")
    annotations = parse_cvat_xml(xml_path)
    print(f"  Found {len(annotations)} annotated images.\n")

    # ── Diagnostic: show all label names found in the XML ─────────────────────
    all_labels_in_xml: set[str] = set()
    for ann in annotations.values():
        all_labels_in_xml.update(ann.keys())
    print(f"\n  Labels found in XML:  {sorted(all_labels_in_xml)}")
    print(f"  Barcode label in use: '{BARCODE_LABEL}'"
          f"  {'✓' if BARCODE_LABEL in all_labels_in_xml else '✗ NOT FOUND — check --barcode-label'}")
    for tl in TEXT_LABELS:
        found = tl in all_labels_in_xml
        print(f"  Text label '{tl}':     {'✓' if found else '✗ NOT FOUND — check --text-labels'}")

    # ── Process each image ────────────────────────────────────────────────────
    all_results = []
    for img_name, annotation in annotations.items():
        print(f"── {img_name}")
        result = process_image(
            img_name=img_name,
            annotation=annotation,
            image_dir=image_dir,
            engines=args.engine,
        )
        all_results.append(result)

        # Optional: save crops for visual inspection
        if args.save_crops and not result.get("error"):
            _save_debug_crops(img_name, annotation, image_dir, output_dir)

    # ── Write outputs ─────────────────────────────────────────────────────────
    json_path = output_dir / "barcode_vs_ocr.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON → {json_path}")

    write_csv(all_results, args.engine, output_dir / "barcode_vs_ocr.csv")

    # ── Print summary + go/no-go ──────────────────────────────────────────────
    print_summary(all_results, args.engine)


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG: optional crop saving
# ─────────────────────────────────────────────────────────────────────────────

def _save_debug_crops(
    img_name: str,
    annotation: dict,
    image_dir: Path,
    output_dir: Path,
) -> None:
    """Save all annotated crops to output_dir/crops/<img_stem>/ for visual QC."""
    stem = Path(img_name).stem
    img_path = None
    for candidate in [
        image_dir / img_name,
        image_dir / (stem + ".jpg"),
        image_dir / (stem + ".jpeg"),
        image_dir / (stem + ".JPG"),
        image_dir / (stem + ".JPEG"),
        image_dir / (stem + ".png"),
        image_dir / (stem + ".tiff"),
    ]:
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        return

    #img = Image.open(img_path).convert("RGB")
    img = Image.open(img_path)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    w, h = img.size
    xml_w = annotation.get("_xml_width")
    xml_h = annotation.get("_xml_height")
    if xml_w and xml_h and (w != xml_w or h != xml_h):
        print(f"  [WARN] {img_name}: XML {xml_w}x{xml_h} vs PIL {w}x{h} — EXIF rotation detected.")

    crop_dir = output_dir / "crops" / stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    annotation_items = list(annotation.items())[2:]
    
    for label, boxes in annotation_items:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box 
            x1c = max(0, x1 - 4)
            y1c = max(0, y1 - 4)
            x2c = min(w, x2 + 4)
            y2c = min(h, y2 + 4)
            crop = img.crop((x1c, y1c, x2c, y2c))
            crop.save(crop_dir / f"{label}_{i}.png")


if __name__ == "__main__":
    main()
