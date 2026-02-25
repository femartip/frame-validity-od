from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class NMSGroup:
    main_box: np.ndarray
    main_score: float
    main_class_probs: np.ndarray
    candidate_boxes: np.ndarray
    candidate_scores: np.ndarray
    candidate_class_probs: np.ndarray


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0.0, inter / union, 0.0)


def group_by_nms_candidates(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    class_probs: np.ndarray,
    iou_thresh: float,
) -> list[NMSGroup]:
    if boxes.size == 0:
        return []

    groups: list[NMSGroup] = []
    classes_unique = np.unique(classes)

    for cls in classes_unique:
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_probs = class_probs[cls_mask]

        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        cls_probs = cls_probs[order]

        remaining = np.arange(len(cls_boxes))
        while remaining.size > 0:
            main_idx = remaining[0]
            main_box = cls_boxes[main_idx]
            main_score = float(cls_scores[main_idx])
            main_probs = cls_probs[main_idx]

            rest = remaining[1:]
            rest_boxes = cls_boxes[rest]
            rest_scores = cls_scores[rest]
            rest_probs = cls_probs[rest]

            ious = _box_iou(main_box, rest_boxes)
            candidate_mask = ious >= iou_thresh

            candidate_boxes = np.concatenate([main_box[None, :], rest_boxes[candidate_mask]], axis=0)
            candidate_scores = np.concatenate([[main_score], rest_scores[candidate_mask]], axis=0)
            candidate_probs = np.concatenate([main_probs[None, :], rest_probs[candidate_mask]], axis=0)

            groups.append(
                NMSGroup(
                    main_box=main_box,
                    main_score=main_score,
                    main_class_probs=main_probs,
                    candidate_boxes=candidate_boxes,
                    candidate_scores=candidate_scores,
                    candidate_class_probs=candidate_probs,
                )
            )

            keep_mask = ~candidate_mask
            remaining = np.concatenate([[main_idx], rest[keep_mask]])
            remaining = remaining[1:]

    return groups


def _box_to_features(box: np.ndarray) -> dict[str, float]:
    x1, y1, x2, y2 = box.tolist()
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    d = w * h
    g = 2.0 * (w + h)
    return {
        "rmin": y1,
        "rmax": y2,
        "cmin": x1,
        "cmax": x2,
        "d": d,
        "g": g,
    }


def compute_metadetect_features_for_detection(group: NMSGroup) -> dict[str, float]:
    main_box = group.main_box
    cand_boxes = group.candidate_boxes
    cand_scores = group.candidate_scores

    main_feats = _box_to_features(main_box)
    cand_rmin = cand_boxes[:, 1]
    cand_rmax = cand_boxes[:, 3]
    cand_cmin = cand_boxes[:, 0]
    cand_cmax = cand_boxes[:, 2]
    cand_w = np.maximum(0.0, cand_boxes[:, 2] - cand_boxes[:, 0])
    cand_h = np.maximum(0.0, cand_boxes[:, 3] - cand_boxes[:, 1])
    cand_d = cand_w * cand_h
    cand_g = 2.0 * (cand_w + cand_h)
    cand_feats_dict = {
        "rmin": cand_rmin,
        "rmax": cand_rmax,
        "cmin": cand_cmin,
        "cmax": cand_cmax,
        "d": cand_d,
        "g": cand_g,
        "s": cand_scores,
    }

    features: dict[str, float] = {}
    features["md_N"] = float(len(cand_boxes))
    features["md_rmin"] = float(main_feats["rmin"])
    features["md_rmax"] = float(main_feats["rmax"])
    features["md_cmin"] = float(main_feats["cmin"])
    features["md_cmax"] = float(main_feats["cmax"])
    features["md_s"] = float(group.main_score)
    features["md_d"] = float(main_feats["d"])
    features["md_g"] = float(main_feats["g"])

    other_boxes = cand_boxes[1:] if len(cand_boxes) > 1 else np.zeros((0, 4), dtype=np.float32)
    other_scores = cand_scores[1:] if len(cand_scores) > 1 else np.zeros((0,), dtype=np.float32)
    if other_boxes.size > 0:
        best_idx = int(np.argmax(other_scores))
        features["md_iou_pb"] = float(_box_iou(main_box, other_boxes[[best_idx]])[0])
    else:
        features["md_iou_pb"] = 0.0

    for name, values in cand_feats_dict.items():
        features[f"md_{name}_cand_min"] = float(np.min(values)) if values.size else 0.0
        features[f"md_{name}_cand_max"] = float(np.max(values)) if values.size else 0.0
        features[f"md_{name}_cand_mean"] = float(np.mean(values)) if values.size else 0.0
        features[f"md_{name}_cand_std"] = float(np.std(values)) if values.size else 0.0

    if other_boxes.size > 0:
        ious = _box_iou(main_box, other_boxes)
        features["md_iou_cand_min"] = float(np.min(ious))
        features["md_iou_cand_max"] = float(np.max(ious))
        features["md_iou_cand_mean"] = float(np.mean(ious))
        features["md_iou_cand_std"] = float(np.std(ious))
    else:
        features["md_iou_cand_min"] = 0.0
        features["md_iou_cand_max"] = 0.0
        features["md_iou_cand_mean"] = 0.0
        features["md_iou_cand_std"] = 0.0

    rd = features["md_d"] / max(features["md_g"], 1e-6)
    features["md_rd"] = float(rd)

    gmin = features["md_g_cand_min"]
    gmax = features["md_g_cand_max"]
    gmean = features["md_g_cand_mean"]
    gstd = features["md_g_cand_std"]

    features["md_rdmin"] = float(features["md_d"] / max(gmin, 1e-6))
    features["md_rdmax"] = float(features["md_d"] / max(gmax, 1e-6))
    features["md_rdmean"] = float(features["md_d"] / max(gmean, 1e-6))
    features["md_rdstd"] = float(features["md_d"] / max(gstd, 1e-6)) if gstd > 0 else 0.0

    for i, prob in enumerate(group.main_class_probs.tolist()):
        features[f"md_p_c{i}"] = float(prob)

    return features


def detection_feature_names(num_classes: int) -> list[str]:
    scalar = [
        "md_N",
        "md_rmin",
        "md_rmax",
        "md_cmin",
        "md_cmax",
        "md_s",
        "md_d",
        "md_g",
        "md_iou_pb",
        "md_rmin_cand_min",
        "md_rmin_cand_max",
        "md_rmin_cand_mean",
        "md_rmin_cand_std",
        "md_rmax_cand_min",
        "md_rmax_cand_max",
        "md_rmax_cand_mean",
        "md_rmax_cand_std",
        "md_cmin_cand_min",
        "md_cmin_cand_max",
        "md_cmin_cand_mean",
        "md_cmin_cand_std",
        "md_cmax_cand_min",
        "md_cmax_cand_max",
        "md_cmax_cand_mean",
        "md_cmax_cand_std",
        "md_s_cand_min",
        "md_s_cand_max",
        "md_s_cand_mean",
        "md_s_cand_std",
        "md_d_cand_min",
        "md_d_cand_max",
        "md_d_cand_mean",
        "md_d_cand_std",
        "md_g_cand_min",
        "md_g_cand_max",
        "md_g_cand_mean",
        "md_g_cand_std",
        "md_iou_cand_min",
        "md_iou_cand_max",
        "md_iou_cand_mean",
        "md_iou_cand_std",
        "md_rd",
        "md_rdmin",
        "md_rdmax",
        "md_rdmean",
        "md_rdstd",
    ]
    class_probs = [f"md_p_c{i}" for i in range(num_classes)]
    return scalar + class_probs


def aggregate_detection_features_to_image(
    detection_features: Iterable[dict[str, float]],
    num_classes: int,
) -> dict[str, float]:
    features_list = list(detection_features)
    feature_names = detection_feature_names(num_classes)

    if not features_list:
        agg: dict[str, float] = {"num_detections": 0.0}
        for name in feature_names:
            if name.startswith("md_p_c"):
                agg[f"{name}_mean"] = 0.0
                agg[f"{name}_max"] = 0.0
            else:
                agg[f"{name}_min"] = 0.0
                agg[f"{name}_max"] = 0.0
                agg[f"{name}_mean"] = 0.0
                agg[f"{name}_std"] = 0.0
        return agg

    agg = {"num_detections": float(len(features_list))}
    for name in feature_names:
        values = np.array([feat.get(name, 0.0) for feat in features_list], dtype=np.float32)
        if name.startswith("md_p_c"):
            agg[f"{name}_mean"] = float(np.mean(values))
            agg[f"{name}_max"] = float(np.max(values))
        else:
            agg[f"{name}_min"] = float(np.min(values))
            agg[f"{name}_max"] = float(np.max(values))
            agg[f"{name}_mean"] = float(np.mean(values))
            agg[f"{name}_std"] = float(np.std(values))
    return agg
