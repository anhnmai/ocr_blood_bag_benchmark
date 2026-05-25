# CVAT annotation guide — blood-bag label images

This document is the labelling protocol for annotators working on the
`blood_bag_ocr_pilot` CVAT project. Follow it precisely so that
`crop_from_cvat.py` can parse the export without manual fixes.

---

## 1. Project setup in CVAT

1. Create a new **task** (not a project) in CVAT: `blood_bag_ocr_pilot`.
2. Task type: **Images** (not video).
3. Upload images from `data/raw/` in batches of 20–30.
4. Set labels exactly as listed in §2 — label names are case-sensitive.

---

## 2. Label names and definitions

| Label | Type | What to draw |
|---|---|---|
| `barcode_region` | Bounding box | Tight box around **each** 1-D barcode stripe. One box per barcode. |
| `donation_id` | Bounding box | The alphanumeric donation identification number (DIN) as printed in text. |
| `blood_group` | Bounding box | The ABO/Rh blood group text (e.g. `A Rh pos`, `0 Rh neg`). |
| `expiry_date` | Bounding box | The expiry/use-by date in `TT.MM.JJJJ` format. |
| `collection_date` | Bounding box | Collection date, if visible. **Mark as skip if not present.** |

**Do not create any other label names.** Do not add attributes.

---

## 3. Box drawing rules

- Draw boxes **tight**: edges should touch the outermost pixel of the text or barcode, plus 1–2 px margin.
- For barcodes: include the full quiet zone (white margin left and right of bars).
- For text fields: include all characters including any surrounding frame line if it cannot be excluded without clipping text.
- If a text field is **partially occluded or unreadable**, still draw the box and add a note in the task description. Do not skip it silently.
- If the same text appears twice (e.g. duplicated donation ID), annotate **both** occurrences with separate boxes.
- Boxes must not overlap unless the underlying regions genuinely overlap in the image.

---

## 4. Quality check before export

For each image, verify:
- [ ] All barcodes on the label have a `barcode_region` box.
- [ ] `donation_id`, `blood_group`, and `expiry_date` each have exactly one box (unless duplicated in the label).
- [ ] No box is drawn for a region that is not present on this label.
- [ ] All boxes are tight and correctly labelled.

---

## 5. Export settings

1. In CVAT, open the task → **Actions → Export dataset**.
2. Format: **CVAT for images 1.1** (XML archive).
3. Save the downloaded `.zip` to `data/cvat_export/`.
4. Unzip so that `data/cvat_export/annotations.xml` exists.
5. Optionally also export as **YOLO 1.1** into `data/yolov8/` for future detection training.

---

## 6. Example label layout

```
┌─────────────────────────────────────────────────────────┐
│  ┌─────────────────────┐  blood_group: [A Rh pos      ] │
│  │ barcode_region      │                                 │
│  │ ▐▌▐▌▌▐▐▌▌▐▌▌▌▐▐▐▌  │  donation_id: [W1234567890   ] │
│  └─────────────────────┘                                 │
│  ┌─────────────────────┐  expiry_date: [31.12.2025     ] │
│  │ barcode_region      │                                 │
│  │ ▐▌▌▐▌▌▐▐▌▌▌▐▌▌▐▐▌  │                                 │
│  └─────────────────────┘                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Common mistakes to avoid

| Mistake | Correct behaviour |
|---|---|
| Using `barcode` instead of `barcode_region` | Use exactly `barcode_region` |
| Drawing one box for the entire label | Draw one box per field |
| Clipping the last character of a donation ID | Expand the box right |
| Annotating a barcode's text interpretation as `donation_id` | Only annotate the printed human-readable text, not the barcode itself |
| Leaving `expiry_date` blank because it is the same as `collection_date` | Both fields are independent; annotate each if present |
