# Frame-Level Trust for Object Detection in Driving Scenes

## Research goal

Build lightweight **assessor models** that predict, for each driving-frame instance, how well a given object detector will perform.

- Primary model `S`: an object detector (e.g., YOLO / Faster R-CNN)
- Meta-features `φ(I)`: context extracted per frame (dataset metadata, weather, image quality, detector outputs)
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

### 4) Run detector inference and compute per-instance IoU/LRP

This produces per-image JSON with keys like `iou`, `lrp`, and confidence stats.

```bash
# YOLO
python src/models/run_inference.py yolo <path-to-yolo-weights.pt> --test

# Faster R-CNN
python src/models/run_inference.py faster-rcnn <path-to-model-weights.pth> --test
```

Outputs (examples):
- `results/yolo/detections.json`
- `results/faster-rcnn/detections.json`

### 4b) (Optional) Save pre-NMS raw predictions for MetaDetect features

```bash
# YOLO raw (pre-NMS)
python src/models/run_inference.py yolo <path-to-yolo-weights.pt> --test --save-raw

# Faster R-CNN raw (pre-NMS)
python src/models/run_inference.py faster-rcnn <path-to-model-weights.pth> --test --save-raw
```

Outputs (examples):
- `results/yolo/raw_predictions.jsonl`
- `results/yolo/raw_predictions.meta.json`
- `results/faster-rcnn/raw_predictions.jsonl`
- `results/faster-rcnn/raw_predictions.meta.json`

You can discretize targets (for classification-style assessors):
```bash
python src/models/run_inference.py yolo <weights.pt> --test --discretize-threshold 0.5
```

### 5) Combine features + targets into a training table

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

### 5a) Upload the tabular datasets to Hugging Face

The repository includes a helper script that uploads the three tabular datasets as separate Hugging Face dataset repositories:

- `data/metafeatures.csv` → ZOD validation meta-features only, no target column
- `data/faster-rcnn_metafeatures.csv` → ZOD validation meta-features + Faster R-CNN targets
- `data/yolo_metafeatures.csv` → ZOD validation meta-features + YOLO targets

For the detector-specific datasets, the target columns are:

- `iou`: mean IoU for the scene
- `lrp`: LRP for the scene

That means downstream users can choose either metric as the prediction target.

Run:

```bash
python src/data/upload_metafeatures_to_hf.py --namespace femartip
```

The script creates separate dataset repos with a short dataset card explaining the source split, feature set, and target columns.

## Hugging Face links

### Models

- Faster R-CNN model: https://huggingface.co/femartip/faster-rcnn-zod
- YOLO model: https://huggingface.co/femartip/yolo-zod

### Datasets

- Metafeatures: https://huggingface.co/datasets/femartip/zod-metafeatures
- Faster R-CNN metafeatures: https://huggingface.co/datasets/femartip/zod-faster-rcnn-metafeatures
- YOLO metafeatures: https://huggingface.co/datasets/femartip/zod-yolo-metafeatures

### 6) Train assessors / analyze results

This part is currently notebook-driven:
- `notebooks/assessors.ipynb`
- `notebooks/assessors_classification.ipynb`
- `notebooks/assess_pred_analysis.ipynb`

### Experiments replicating MetaDetect, on metadetect branch

```bash
# YOLO MetaDetect
python src/data/build_metadetect_dataset.py yolo --raw results/yolo/raw_predictions.jsonl --targets results/yolo/detections.json

# Faster R-CNN MetaDetect
python src/data/build_metadetect_dataset.py faster-rcnn --raw results/faster-rcnn/raw_predictions.jsonl --targets results/faster-rcnn/detections.json
```

Outputs (examples):
- `data/yolo_metadetect.csv`
- `data/faster-rcnn_metadetect.csv`

## Lessons learned

- Adding the image as input does not help.
- Using an MLLM to extract features also does not help.
- Fine-tuning a realtively small MLLM to directly predict the validity indicator from the image, also does not help.  

