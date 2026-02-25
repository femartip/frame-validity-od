import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metadetect_features import (
    aggregate_detection_features_to_image,
    compute_metadetect_features_for_detection,
    detection_feature_names,
    group_by_nms_candidates,
)

# Hard cap to keep pre-NMS candidate counts tractable per class.
# Tune if needed; this is applied before grouping.
MAX_PRE_NMS_PER_CLASS = 1000


def _load_raw_predictions(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.suffix == ".jsonl":
        preds: dict[str, Any] = {}
        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                image_id = str(rec.pop("image_id"))
                preds[image_id] = rec
        meta_path = path.with_suffix(".meta.json")
        metadata = {}
        if meta_path.exists():
            with meta_path.open("r") as f:
                metadata = json.load(f)
        return preds, metadata

    with path.open("r") as f:
        raw = json.load(f)
    if "predictions" in raw:
        return raw.get("predictions", {}), raw.get("metadata", {})
    return raw, {}


def _get_target(targets: dict[str, Any], image_id: str) -> dict[str, Any] | None:
    if image_id in targets:
        return targets[image_id]
    if str(image_id) in targets:
        return targets[str(image_id)]
    try:
        int_id = str(int(image_id))
    except (TypeError, ValueError):
        return None
    return targets.get(int_id)


def _aggregate_columns(num_classes: int) -> list[str]:
    cols = ["num_detections"]
    for name in detection_feature_names(num_classes):
        if name.startswith("md_p_c"):
            cols.append(f"{name}_mean")
            cols.append(f"{name}_max")
        else:
            cols.append(f"{name}_min")
            cols.append(f"{name}_max")
            cols.append(f"{name}_mean")
            cols.append(f"{name}_std")
    cols.extend(["iou", "lrp"])
    return cols


def _compute_agg_from_pred(pred: dict[str, Any] | None, nms_iou: float, num_classes: int) -> dict[str, float]:
    if pred is None or len(pred.get("boxes", [])) == 0:
        return aggregate_detection_features_to_image([], num_classes=num_classes)

    boxes = np.asarray(pred["boxes"], dtype=np.float32)
    scores = np.asarray(pred["scores"], dtype=np.float32)
    class_probs = np.asarray(pred["class_probs"], dtype=np.float32)
    if class_probs.ndim == 1:
        class_probs = class_probs.reshape(-1, num_classes)
    classes = pred.get("classes")
    if classes is None:
        classes = class_probs.argmax(axis=1)
    classes = np.asarray(classes, dtype=np.int64)

    if MAX_PRE_NMS_PER_CLASS is not None:
        keep_indices = []
        for cls in np.unique(classes):
            cls_idx = np.where(classes == cls)[0]
            if cls_idx.size > MAX_PRE_NMS_PER_CLASS:
                cls_scores = scores[cls_idx]
                topk = np.argpartition(-cls_scores, MAX_PRE_NMS_PER_CLASS - 1)[:MAX_PRE_NMS_PER_CLASS]
                cls_idx = cls_idx[topk]
            keep_indices.append(cls_idx)
        keep = np.concatenate(keep_indices) if keep_indices else np.array([], dtype=np.int64)
        boxes = boxes[keep]
        scores = scores[keep]
        class_probs = class_probs[keep]
        classes = classes[keep]

    groups = group_by_nms_candidates(
        boxes=boxes,
        scores=scores,
        classes=classes,
        class_probs=class_probs,
        iou_thresh=float(nms_iou),
    )
    det_features = [compute_metadetect_features_for_detection(g) for g in groups]
    return aggregate_detection_features_to_image(det_features, num_classes=num_classes)


def _stream_jsonl_predictions(path: Path):
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            image_id = str(rec.pop("image_id"))
            yield image_id, rec


def _load_metadata_for_jsonl(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        with meta_path.open("r") as f:
            return json.load(f)
    return {}


def build_dataset(
    raw_path: Path,
    targets_path: Path,
    nms_iou: float | None,
    num_classes: int | None,
) -> pd.DataFrame:
    raw_preds, meta = _load_raw_predictions(raw_path)
    with targets_path.open("r") as f:
        targets = json.load(f)

    if nms_iou is None:
        nms_iou = float(meta.get("nms_iou", 0.5))
    if num_classes is None:
        num_classes = meta.get("num_classes")

    if num_classes is None:
        for pred in raw_preds.values():
            probs = pred.get("class_probs")
            if probs:
                num_classes = len(probs[0])
                break
    if num_classes is None:
        raise ValueError("Unable to infer num_classes. Provide --num-classes.")

    rows: dict[str, dict[str, float]] = {}
    for image_id, target in targets.items():
        pred = raw_preds.get(image_id)
        if pred is None:
            pred = raw_preds.get(str(image_id))

        agg = _compute_agg_from_pred(pred, nms_iou=float(nms_iou), num_classes=num_classes)
        agg["iou"] = float(target.get("iou", 0.0)) if target.get("iou") is not None else None
        agg["lrp"] = float(target.get("lrp", 0.0)) if target.get("lrp") is not None else None
        rows[str(image_id)] = agg

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df


def build_dataset_streaming(
    raw_path: Path,
    targets_path: Path,
    out_path: Path,
    nms_iou: float | None,
    num_classes: int | None,
) -> None:
    with targets_path.open("r") as f:
        targets = json.load(f)

    meta = _load_metadata_for_jsonl(raw_path)
    if nms_iou is None:
        nms_iou = float(meta.get("nms_iou", 0.5))
    if num_classes is None:
        num_classes = meta.get("num_classes")

    if num_classes is None:
        for _, pred in _stream_jsonl_predictions(raw_path):
            probs = pred.get("class_probs")
            if probs:
                num_classes = len(probs[0])
                break
        if num_classes is None:
            raise ValueError("Unable to infer num_classes. Provide --num-classes.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _aggregate_columns(num_classes)
    with out_path.open("w") as f:
        f.write(",".join(["image_id"] + columns) + "\n")
        for image_id, pred in _stream_jsonl_predictions(raw_path):
            target = _get_target(targets, image_id)
            if target is None:
                continue
            agg = _compute_agg_from_pred(pred, nms_iou=float(nms_iou), num_classes=num_classes)
            agg["iou"] = float(target.get("iou", 0.0)) if target.get("iou") is not None else None
            agg["lrp"] = float(target.get("lrp", 0.0)) if target.get("lrp") is not None else None
            row = [image_id] + [str(agg.get(col, 0.0)) for col in columns]
            f.write(",".join(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name (e.g., yolo, faster-rcnn)")
    parser.add_argument("--raw", required=True, help="Path to raw_predictions.json")
    parser.add_argument("--targets", required=True, help="Path to detections.json")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--nms-iou", type=float, default=None, help="Override NMS IoU threshold for grouping")
    parser.add_argument("--num-classes", type=int, default=None, help="Override number of classes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_path = Path(args.raw)
    targets_path = Path(args.targets)
    out_path = Path(args.out) if args.out else Path("data") / f"{args.model}_metadetect.csv"

    if raw_path.suffix == ".jsonl":
        build_dataset_streaming(
            raw_path,
            targets_path,
            out_path,
            nms_iou=args.nms_iou,
            num_classes=args.num_classes,
        )
    else:
        df = build_dataset(raw_path, targets_path, nms_iou=args.nms_iou, num_classes=args.num_classes)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=True)

    print(f"Saved MetaDetect dataset to {out_path}")
