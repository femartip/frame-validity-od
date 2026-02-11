# Predictability-AD

Research code for **instance-level predictability / performance assessment of object detection systems** in driving scenes.

## Research goal

Build lightweight **assessor models** that predict, for each driving-frame instance, how well a given object detector will perform.

- Primary model `S`: an object detector (e.g., YOLO / Faster R-CNN / RF-DETR)
- Meta-features `φ(I)`: context extracted per frame (dataset metadata, weather, image quality, embeddings, detector outputs)
- Target `V(I,S)`: an instance-level validity indicator (e.g., meanIoU-with-zeros, LRP)
- Assessor `M`: a meta-model trained so that `M(φ(I), S) ≈ V(I,S)`

## Repo layout

- `src/data/`
  - feature extraction and dataset building
- `src/models/`
  - detector inference + per-instance metric computation
- `src/utils/`
  - conversion utilities (ZOD → COCO → YOLO)
- `notebooks/`
  - analysis + assessor training notebooks
- `data/`
  - cached CSVs / artifacts used by scripts and notebooks
- `results/`
  - detections JSONs, tables, plots

## Installation

Python project defined in `pyproject.toml`.

### Poetry

```bash
cd Predictability-AD
poetry install
poetry shell
```

Detectron2 needs to be installed separetly, can use: pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+fd27788pt2.9.1cu128

### Notes (important)

- Python: `>=3.11, <3.14`
- Some deps are heavy and/or pinned (CUDA torch, git deps such as ZOD and RF-DETR). If you don’t have CUDA, expect to adjust `torch/torchvision` pins.

## Data expectations / local layout

This project assumes data is placed under `./data/`.

### ZOD Frames

- Place ZOD under:
  - `data/zod/`

Scripts use:
- `ZodFrames(dataset_root="./data/zod", version="full")`

### COCO and YOLO derived datasets

The repo includes scripts to generate:
- COCO annotations: `data/zod_coco/*.json`
- YOLO dataset: `data/zod_yolo/images/{train,val,test}/` and `data/zod_yolo/labels/{train,val,test}/`

## End-to-end pipeline (recommended)

This is the minimal sequence that matches the current codebase.

### 1) Convert ZOD → COCO

```bash
python src/utils/zod_to_coco.py
```

This writes (example names):
- `data/zod_coco/zod_full_Anonymization.BLUR_train.json`
- `data/zod_coco/zod_full_Anonymization.BLUR_val.json`
- `data/zod_coco/zod_full_Anonymization.BLUR_test.json`

### 2) Convert COCO → YOLO (and copy images)

```bash
python src/utils/coco_to_yolo.py
```

This creates:
- `data/zod_yolo/images/...`
- `data/zod_yolo/labels/...`
- `data/zod_yolo/dataset.yaml`

### 3) Build tabular meta-features (metadata + weather + image quality)

```bash
python src/data/zod_to_tabular.py 1000
# add --resume to append without recomputing existing ids
python src/data/zod_to_tabular.py 1000 --resume
```

Output:
- `data/metafeatures.csv`

### 4) (Optional) LLM-derived meta-features

Requires a `.env` in repo root with keys referenced by the script:
- `OPENAI_KEY`
- `GOOGLE_API_KEY`

Run:
```bash
python src/data/llm_feature_extraction.py openai
# or
python src/data/llm_feature_extraction.py gemini

# resume mode reuses existing description/csv if present
python src/data/llm_feature_extraction.py openai --resume
```

Outputs:
- `data/llm_metafeatures_description.json`
- `data/llm_metafeatures.csv`

### 5) Run detector inference and compute per-instance IoU/LRP

This produces per-image JSON with keys like `iou`, `lrp`, and confidence stats.

```bash
# YOLO
python src/models/run_inference.py yolo <path-to-yolo-weights.pt> --test

# Faster R-CNN
python src/models/run_inference.py faster-rcnn <path-to-model-weights.pth> --test

# RF-DETR
python src/models/run_inference.py rf-detr <path-to-rfdetr-weights> --test
```

Outputs (examples):
- `results/yolo/detections.json`
- `results/faster-rcnn/detections.json`
- `results/rf-detr/detections.json`

You can discretize targets (for classification-style assessors):
```bash
python src/models/run_inference.py yolo <weights.pt> --test --discretize-threshold 0.5
```

### 6) Combine features + targets into a training table

```bash
# Use hand-crafted meta-features
python src/data/combine_data_predictions.py yolo metafeatures

# Use LLM meta-features
python src/data/combine_data_predictions.py yolo llm-metafeatures

# Use discretized detections
python src/data/combine_data_predictions.py yolo metafeatures --discretize
```

Outputs (examples):
- `data/yolo_metafeatures.csv`
- `data/yolo_llm-metafeatures.csv`
- `data/yolo_metafeatures_disc.csv`

### 7) Train assessors / analyze results

This part is currently notebook-driven:
- `notebooks/assessors.ipynb`
- `notebooks/assessors_classification.ipynb`
- `notebooks/assess_pred_analysis.ipynb`

## Results (current, from the draft)

You already have an initial set of assessor results in the Obsidian paper draft.

At a high level:
- baselines using detector confidence already explain a chunk of variance,
- richer meta-features improve performance modestly,
- AutoGluon/ensembles tend to be strongest.

For the full table/plots, see:
- Obsidian: `PhD/01 Predictability - OD/Paper.md`
- Repo outputs: `results/`
