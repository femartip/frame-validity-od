# Predictability-AD — MetaDetect Pipeline

This repo is configured to run the **MetaDetect** pipeline on the ZOD dataset and train **image-level** assessor models that predict detection quality (IoU / LRP). The pipeline extracts **model-intrinsic** features from **pre-NMS predictions**, groups NMS candidates per detection, computes MetaDetect features, and aggregates them per image for assessor training.

This README only documents the MetaDetect pipeline.

## What MetaDetect Does (Conceptually)

MetaDetect builds features from a detector’s **raw pre-NMS predictions**. For each detection, it uses:
- Box geometry and score
- Class probabilities
- Statistics over **NMS candidate boxes** (boxes suppressed by that detection)
- IoU statistics between the detection and its candidates

These per-detection features are then **aggregated per image** (min/max/mean/std for scalars; mean/max for class probabilities) so they can be used with the image-level assessor workflow in this repo.

## How This Differs From the Original Paper

The original MetaDetect setup is **per-detection**. This repo uses **image-level aggregation** so it can plug into existing assessor training and evaluation. Key differences:

- **Granularity:**
  - Paper: per-detection assessors
  - Here: per-image aggregated features
- **Candidate sets:**
  - Paper: exact NMS-suppressed candidates per detection
  - Here: reconstructed candidate grouping using the model’s NMS IoU threshold (same greedy logic)
- **MC-dropout features:**
  - Paper: optional 66+C feature set
  - Here: not used

If you need paper-faithful results, remove aggregation and train per-detection assessors.

## Requirements

- Python >= 3.11
- ZOD dataset under `./data/zod/`
- Detectron2 + Ultralytics installed for Faster R-CNN / YOLO

## Pipeline (MetaDetect)

### 1) Prepare ZOD in YOLO / COCO format

```bash
# ZOD → COCO
python src/utils/zod_to_coco.py

# COCO → YOLO
python src/utils/coco_to_yolo.py
```

### 2) Run detector inference and save **pre-NMS** raw predictions

```bash
# YOLO (pre-NMS raw + detections)
python src/models/run_inference.py yolo <path-to-yolo-weights.pt> --test --save-raw

# Faster R-CNN (pre-NMS raw + detections)
python src/models/run_inference.py faster-rcnn <path-to-model-weights.pth> --test --save-raw
```

Outputs:
- `results/<model>/detections.json`
- `results/<model>/raw_predictions.jsonl`
- `results/<model>/raw_predictions.meta.json`

### 3) Build the MetaDetect dataset (image-level)

```bash
python src/data/build_metadetect_dataset.py yolo --raw results/yolo/raw_predictions.jsonl --targets results/yolo/detections.json
python src/data/build_metadetect_dataset.py faster-rcnn --raw results/faster-rcnn/raw_predictions.jsonl --targets results/faster-rcnn/detections.json
```

Outputs:
- `data/yolo_metadetect.csv`
- `data/faster-rcnn_metadetect.csv`

### 4) Train assessors

Use the MetaDetect assessor notebook (image-level):

- `notebooks/assessors_metadetect.ipynb`

This notebook trains multiple regressors (Linear / RF / MLP / XGBoost / AutoGluon) for IoU and LRP.

## Notes on Performance

If `build_metadetect_dataset.py` is slow, the cause is typically very large pre-NMS candidate counts. The script caps pre-NMS candidates per class (default 1000) to keep runtime tractable.

You can adjust this cap in `src/data/build_metadetect_dataset.py`:
```python
MAX_PRE_NMS_PER_CLASS = 1000
```

## Outputs You Should Expect

- Image-level MetaDetect feature CSVs
- Assessor performance tables and plots from the notebook

