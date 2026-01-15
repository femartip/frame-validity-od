import argparse
import os
import os.path as osp
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor
import metrics

from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo


def load_model(checkpoint_path: str) -> DefaultPredictor:
    cfg = get_cfg()    # obtain detectron2's default config
    cfg.merge_from_file("./models/faster-rcnn/config.yaml")   # load values from a file    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()
    return DefaultPredictor(cfg)


def annotations_from_coco(dataset_dict: dict) -> list[dict]:
    boxes = []
    labels = []
    for ann in dataset_dict.get("annotations", []):
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(int(ann["category_id"]))
    return [{
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }]


def process_image(dataset_dict: dict, predictor: DefaultPredictor) -> dict:
    image_path = dataset_dict["file_name"]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return {"iou": None, "lrp": None, "confidence": []}

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    labels = instances.pred_classes

    detections = [{
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }]
    annotations = annotations_from_coco(dataset_dict)

    if detections[0]["boxes"].numel() == 0:
        return {"iou": 0.0, "lrp": 0.0, "confidence": []}
    if annotations[0]["boxes"].numel() == 0:
        print(f"WARNING: No annotations found for image {image_path}.")
        return {"iou": None, "lrp": None, "confidence": []}

    metrics_dict = metrics.get_metrics(detections, annotations, metrics=["iou", "lrp"])
    metrics_dict["confidence"] = scores.tolist()
    return metrics_dict


def save_to_json(all_detections: dict, output_folder: str, filename: str) -> None:
    json_output_path = os.path.join(output_folder, filename)
    with open(json_output_path, "w") as json_file:
        json.dump(all_detections, json_file, indent=4)
    print(f"All detections saved to {json_output_path}")
    print(f"Total images processed: {len(all_detections)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--zod-path", default="./data/zod_coco/", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--image-root", default=".", type=str)
    parser.add_argument("--output-dir", default="results/detectron2", type=str)
    args = parser.parse_args()

    json_coco = "_annotations.coco.json"

    if args.train:
        register_coco_instances("train", {}, osp.join(args.zod_path, "train", json_coco), args.image_root)
        train_dataset = DatasetCatalog.get("train")

    if args.val:
        register_coco_instances("valid", {}, osp.join(args.zod_path, "valid", json_coco), args.image_root)
        val_dataset = DatasetCatalog.get("valid")

    os.makedirs(args.output_dir, exist_ok=True)
    predictor = load_model(args.checkpoint)

    if args.train:
        print(f"Processing {len(train_dataset)} training images")
        iou_vals = []
        lrp_vals = []
        for d in tqdm(train_dataset):
            results = process_image(d, predictor)
            iou_val = results.get("iou", 0.0)
            lrp_val = results.get("lrp", 0.0)
            if iou_val is None:
                continue
            iou_vals.append(iou_val)
            lrp_vals.append(lrp_val)
        print(f"Training Mean IoU: {np.mean(iou_vals):.4f}")
        print(f"Training Mean LRP: {np.mean(lrp_vals):.4f}")

    if args.val:
        print(f"Processing {len(val_dataset)} validation images")
        detections_val = {}
        iou_vals = []
        lrp_vals = []
        for d in tqdm(val_dataset):
            results = process_image(d, predictor)
            iou_val = results.get("iou", 0.0)
            lrp_val = results.get("lrp", 0.0)
            if iou_val is None:
                continue
            iou_vals.append(iou_val)
            lrp_vals.append(lrp_val)
            image_id = str(d.get("image_id", osp.basename(d["file_name"])))
            detections_val[image_id] = results

        print(f"Validation Mean IoU: {np.mean(iou_vals):.4f}")
        print(f"Validation Mean LRP: {np.mean(lrp_vals):.4f}")
        save_to_json(detections_val, args.output_dir, "detections.json")


if __name__ == "__main__":
    main()
