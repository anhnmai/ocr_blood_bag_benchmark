# ocr_blood_bag_benchmark

> **Edge-aware OCR benchmarking for German blood-bag labels**  
> On-going project 

This repository contains the full pipeline of the study:  
a reproducible, two-axis benchmark (OCR accuracy × on-device inference cost)  
of multiple OCR engines on German blood-bag labels conforming to  
ISBT 128 / DIN 58905, with barcode-decoded ground truth as the evaluation oracle.

---

## Project overview

```
Phase 0 (Done): A small-scale Feasibility test → 
Phase 1 (tbu): Dataset + protocol      →  data/  + config/cfg.yaml
Phase 2 (tbu):  Zero-shot benchmark     →  results/benchmark_zero_shot.csv
Phase 3 (tbu):  Ablation (constraint,   →  results/ablation_results.csv
         fine-tuning, both)
Phase 4 (tbu):  Stress test + battery   →  results/stress_test_results.csv
                                     results/battery_proxy.json
```

---

## Repository structure

```
ocr_blood_bag_benchmark/
├── data/
│   ├── blood_bag_pilot_small    # small scale dataset for feasibility test 
│   │   ├──raw/                     # original images (not committed; add via DVC or LFS)
│   │   ├──cvat_export/             # CVAT XML exports
│   │   └──yolov8/                  # optional YOLO format export for detection
├── src/
│   ├── data_prep/
│   │   └── crop_from_cvat.py    # parse CVAT XML → cropped field images
│   ├── barcodes/
│   │   ├── barcode_vs_ocr.py    # FEASIBILITY SCRIPT: crop→decode→OCR→compare
│   │   └── decode_barcodes.py   # ZBar barcode decoder → GT JSON
│   ├── metrics/
│   │   └── calculate_cer_wer.py # CER / WER metrics (Levenshtein-based)
│   └── mobile/
│       └── stress_test.py       # degradation + battery proxy tests
├── models/                      # fine-tuned model weights (not committed)
│   ├── tesseract_finetuned/
│   └── paddleocr_finetuned/
├── results/                     # auto-generated CSVs and plots
├── config/
│   └── cfg.yaml                 # master configuration — edit before running
├── docs/
│   └── annotation_guide.md      # CVAT labelling protocol
├── tests/
├── requirements.txt
└── README.md
```

---

## Feasibility test (Done)

Before running the full pipeline, validate the barcode-as-oracle principle
on 20–30 annotated images with a single command:

```bash
python src/barcodes/barcode_vs_ocr.py \
    --annotations data/cvat_export/annotations.xml \
    --images      data/raw/ \
    --engine      tesseract \
    --save_crops          # optional: saves debug crops for visual QC
```

Add `--engine tesseract easyocr` to compare both engines in one run.

The script:
1. Crops every `barcode_region` box from each image.
2. Decodes each barcode with **pyzbar** (ZBar), trying multiple preprocessing
   variants to maximise decode success on mobile-camera shots.
3. Parses the raw barcode payload into ISBT 128 fields
   (`donation_id`, `blood_group`, `expiry_date`).
4. Crops the corresponding text field boxes and runs OCR.
5. Computes CER between barcode-decoded value and OCR-read value per field.
6. Prints a go/no-go summary table with decode rate, mean CER, and p95 latency.

Output files:
- `results/barcode_vs_ocr.json` — full per-image detail
- `results/barcode_vs_ocr.csv`  — one row per (image, engine, field)
- `results/crops/<stem>/`       — debug crops (if `--save_crops`)

**Go/no-go criteria:**

| Signal | Threshold | Decision |
|---|---|---|
| Barcode decode rate | ≥ 80 % | Proceed |
| Mean CER (donation_id) | < 0.10 | Engine is viable |
| p95 latency (laptop) | < 200 ms | Safe headroom for ARM |

---

## Full pipeline quick start 

### 1. Install dependencies (tbu)


### 2. Annotate images in CVAT

See `docs/annotation_guide.md` for the full labelling protocol.  
Export as **CVAT for images (XML 1.1)** → save to `data/cvat_export/`.

### 3. Crop field regions

```bash
python src/data_prep/crop_from_cvat.py \
    --annotations data/cvat_export/annotations.xml \
    --images      data/raw/ \
    --output      data/crops/
```

### 4. Decode barcodes (ground truth)

```bash
python src/barcodes/decode_barcodes.py \
    --crops   data/crops/barcode_region/ \
    --output  data/barcodes/ground_truth.json
```

### 5. Run zero-shot benchmark

```bash
python src/ocr/run_ocr_benchmark.py \
    --crops  data/crops/ \
    --gt     data/barcodes/ground_truth.json \
    --output results/benchmark_zero_shot.csv
```

### 6. Run ablation study (tbu)


### 7. Stress test (tbu)

```bash
python src/mobile/stress_test.py \
    --crops   data/crops/ \
    --gt      data/barcodes/ground_truth.json \
    --engine  tesseract
```

---

## Evaluation metrics

| Metric | Definition |
|---|---|
| CER | Character Error Rate = Levenshtein distance / reference length [14] |
| WER | Word Error Rate = token-level Levenshtein / reference word count |
| p95 latency | 95th-percentile inference time over 20 runs on the reference device [15] |
| RAM Δ | Peak RSS delta between engine load and first inference |
| Model size | On-disk size of model weights in MB |

---

## OCR engines evaluated



---

## Ablation conditions


---

## References

```
[10] B. Sekachev et al., "Computer Vision Annotation Tool: A Universal Approach
     to Data Annotation," 2019. [Online]. Available: https://github.com/cvat-ai/cvat

[11] R. Smith, "An Overview of the Tesseract OCR Engine," in Proc. 9th IAPR Int.
     Conf. Document Analysis and Recognition (ICDAR), 2007, pp. 629–633.

[12] PaddlePaddle Authors, "PaddleOCR: Awesome Multilingual OCR Toolkits Based on
     PaddlePaddle," 2020. [Online]. Available: https://github.com/PaddlePaddle/PaddleOCR

[13] J. H. Baek et al., "What Is Wrong With Scene Text Recognition Model
     Comparisons? Dataset and Model Analysis," in Proc. ICCV, 2019, pp. 4715–4723.
     [EasyOCR builds on CRAFT + CRNN architectures benchmarked here.]

[14] V. I. Levenshtein, "Binary codes capable of correcting deletions, insertions
     and reversals," Soviet Physics Doklady, vol. 10, no. 8, pp. 707–710, 1966.

[15] J. Hou et al., "MNN: A universal and efficient inference engine," in
     Proc. 3rd MLSys Conf., Austin, TX, USA, 2020, pp. 1–13.

[16] P. Garst, R. Ingle, and Y. Fujii, "OCR language models with custom
     vocabularies," in Document Analysis and Recognition – ICDAR 2023,
     LNCS vol. 14190. Cham: Springer, 2023, pp. 105–120.
     DOI: 10.1007/978-3-031-41685-9_7
```

---

## Standards

- **ISBT 128**: International standard for labelling of blood components.  
  Reference: ISBT, *ISBT 128 for Blood Components: An Introduction*, v8.
- **DIN 58905**: German national standard for blood bag labelling notation.
