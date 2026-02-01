# Predictability-AD

Research code for **instance-level predictability / performance assessment of object detection systems** in driving scenes.

This project supports the workflow described in the Obsidian PhD notes:
- Obsidian (paper + research log): `~/Documents/Obsedian/PhD/01 Predictability - OD/`
- This repo (code + notebooks): `~/Documents/Github/Predictability-AD/`

## Research goal (high level)

Build lightweight **assessor models** that predict, for each driving-frame instance, how well a given object detector will perform.

Concretely:
- Primary model `S`: an object detector (e.g., YOLO / Faster R-CNN / RF-DETR)
- Meta-features `φ(I)`: context extracted from each frame (metadata, weather, image quality, system outputs, embeddings)
- Target `V(I,S)`: an instance-level validity indicator (e.g., meanIoU-with-zeros, LRP)
- Assessor `M`: a meta-model trained so that `M(φ(I), S) ≈ V(I,S)`

The working paper draft lives in Obsidian:
- `PhD/01 Predictability - OD/Paper.md`

## Repo layout

- `src/data/`
  - data prep and feature extraction (tabular meta-features, LLM feature extraction, embeddings)
- `src/models/`
  - detector training / inference helpers and metric computation
- `src/utils/`
  - dataset conversion utilities (COCO/YOLO/ZOD) + BRISQUE utilities
- `notebooks/`
  - EDA, dataset splits, assessor training and analysis
- `data/`
  - cached CSVs used by notebooks/scripts (e.g., metafeatures, embeddings)
- `results/`
  - outputs, plots, tables

## Environment / installation

This repo is configured as a Python project via `pyproject.toml`.

### Option 1: Poetry

```bash
cd Predictability-AD
poetry install
poetry shell
```

### Option 2: pip (less reproducible)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Notes

- Python: `>=3.11, <3.14`
- The project depends on a CUDA-specific PyTorch index (`cu128`). If you don’t have CUDA, you may need to adapt the torch/torchvision pins.
- Some dependencies pull from git (e.g. ZOD, RF-DETR).

## Data

Experiments reference **Zenseact Open Dataset (ZOD) Frames**.

You’ll need to obtain/configure access separately (see the notebooks that use `zod` tooling).

## Typical workflow

1) Create the tabular dataset / meta-features
- see `src/data/zod_to_tabular.py` and the notebooks `notebooks/zod_to_tabular.ipynb`

2) Train / run detectors + compute per-instance targets
- see `src/models/run_inference.py`

3) Train assessors
- see notebooks:
  - `notebooks/assessors.ipynb`
  - `notebooks/assessors_classification.ipynb`
  - `notebooks/assess_pred_analysis.ipynb`

4) Write up results
- Obsidian draft: `PhD/01 Predictability - OD/Paper.md`

## Outputs

- Intermediate CSVs are kept under `data/`
- Experimental outputs/plots under `results/`

## Citation

If you reuse this for your own work, cite the related paper/draft (once public). For now, this repo is research-in-progress.
