import argparse
import json
import os
from pathlib import Path

import metrics

import cv2
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from rfdetr import RFDETRBase       #type: ignore

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

from ultralytics import YOLO    #type: ignore


def load_yolo_annotations(annotation_path: str, img_width: int, img_height: int) -> list[dict]:
    boxes = []
    labels = []
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    x_min = xc - w / 2
                    y_min = yc - h / 2
                    x_max = xc + w / 2
                    y_max = yc + h / 2
                    x1 = int(x_min * img_width)
                    y1 = int(y_min * img_height)
                    x2 = int(x_max * img_width)
                    y2 = int(y_max * img_height)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
    return [{"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.long)}]


def list_images(image_root: str) -> list[str]:
    image_extensions = [".jpg", ".png", ".jpeg"]
    files = [f for f in os.listdir(image_root) if any(f.lower().endswith(ext) for ext in image_extensions)]
    files.sort()
    return files


def discretize_metrics(metrics_dict: dict, threshold: float) -> dict:
    for key in ("iou", "lrp"):
        value = metrics_dict.get(key)
        if value is None:
            continue
        metrics_dict[key] = 1.0 if value > threshold else 0.0
    return metrics_dict


def save_iou0_image(image, pred_boxes, gt_boxes, iou, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    img = image.copy()
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy().tolist()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy().tolist()
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"IoU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_iou0.jpg"), img)


def save_predictions(predictions_dict, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, filename)
    with open(json_output_path, "w") as json_file:
        json.dump(predictions_dict, json_file, indent=4)
    print(f"Predictions saved to {json_output_path}")


def run_yolo(model: YOLO, image_root: str, output_dir: str, output_name: str | None = None, discretize_threshold: float | None = None, save_zero_iou: bool = False, save_preds: bool = False) -> None:
    detections_val = {}
    predictions_val = {}
    iou_vals = []
    lrp_vals = []
    image_files = list_images(image_root)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_root, image_file)

        #Process image logic
        annotation_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        image = cv2.imread(image_path)
        
        img_height, img_width = image.shape[:2]
        annotations = load_yolo_annotations(annotation_path, img_width, img_height)

        results = model(image, verbose=False)[0]
        boxes = [box.xyxy.cpu().numpy()[0].tolist() for box in results.boxes]
        boxes_conf = [box.conf.cpu().numpy()[0].item() for box in results.boxes]
        boxes_cls = [box.cls.cpu().numpy()[0].item() for box in results.boxes]
        detections = [{
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "scores": torch.tensor(boxes_conf, dtype=torch.float32),
            "labels": torch.tensor(boxes_cls, dtype=torch.long),
        }]

        if detections[0]["boxes"].numel() == 0:
            metrics_dict = {"iou": 0.0, "lrp": 0.0, "confidence": []}
        elif annotations[0]["boxes"].numel() == 0:
            print(f"WARNING: No annotations found for image {image_path}.")
            metrics_dict = {"iou": None, "lrp": None, "confidence": []}
        else:
            metrics_dict = metrics.get_metrics(detections, annotations, metrics=["iou", "lrp"])
            metrics_dict["confidence"] = boxes_conf

        if save_zero_iou and metrics_dict.get("iou") == 0.0:
            image_name = os.path.splitext(image_file)[0]
            save_iou0_image(image, detections[0]["boxes"], annotations[0]["boxes"], metrics_dict["iou"], os.path.join(output_dir, "iou0_images"), image_name)

        if discretize_threshold != None:
            metrics_dict = discretize_metrics(metrics_dict, discretize_threshold)

        iou_val = metrics_dict.get("iou", 0.0)
        lrp_val = metrics_dict.get("lrp", 0.0)
        if iou_val is not None:
            iou_vals.append(iou_val)
            lrp_vals.append(lrp_val)
        image_id = image_file.split("_")[0]
        detections_val[image_id] = metrics_dict
        if save_preds:
            predictions_val[image_id] = {
                "boxes": boxes,
                "scores": boxes_conf,
                "labels": boxes_cls,
            }

    print(f"Mean IoU: {np.mean(iou_vals):.4f}")
    print(f"Mean LRP: {np.mean(lrp_vals):.4f}")

    if output_name != None:
        save_to_json(detections_val, output_dir, output_name)
    if save_preds:
        save_predictions(predictions_val, output_dir, "predictions.json")


def run_rf_detr(model: RFDETRBase, image_root: str, output_dir: str, output_name: str | None = None, discretize_threshold: float | None = None, save_zero_iou: bool = False, save_preds: bool = False) -> None:
    detections_val = {}
    predictions_val = {}
    iou_vals = []
    lrp_vals = []
    image_files = list_images(image_root)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_root, image_file)
        annotation_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}.")
            continue
        img_height, img_width = image.shape[:2]
        annotations = load_yolo_annotations(annotation_path, img_width, img_height)

        results = model.predict(image)[0]
        boxes = [box.xyxy.cpu().numpy()[0].tolist() for box in results.boxes]    #type: ignore
        boxes_conf = [box.conf.cpu().numpy()[0].item() for box in results.boxes]    #type: ignore
        boxes_cls = [box.cls.cpu().numpy()[0].item() for box in results.boxes]  #type: ignore
        detections = [{
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "scores": torch.tensor(boxes_conf, dtype=torch.float32),
            "labels": torch.tensor(boxes_cls, dtype=torch.long),
        }]

        if detections[0]["boxes"].numel() == 0:
            metrics_dict = {"iou": 0.0, "lrp": 0.0, "confidence": []}
        elif annotations[0]["boxes"].numel() == 0:
            print(f"WARNING: No annotations found for image {image_path}.")
            metrics_dict = {"iou": None, "lrp": None, "confidence": []}
        else:
            metrics_dict = metrics.get_metrics(detections, annotations, metrics=["iou", "lrp"])
            metrics_dict["confidence"] = boxes_conf

        if save_zero_iou and metrics_dict.get("iou") == 0.0:
            image_name = os.path.splitext(image_file)[0]
            save_iou0_image(image, detections[0]["boxes"], annotations[0]["boxes"], metrics_dict["iou"], os.path.join(output_dir, "iou0_images"), image_name)

        if discretize_threshold != None:
            metrics_dict = discretize_metrics(metrics_dict, discretize_threshold)

        iou_val = metrics_dict.get("iou", 0.0)
        lrp_val = metrics_dict.get("lrp", 0.0)
        if iou_val is not None:
            iou_vals.append(iou_val)
            lrp_vals.append(lrp_val)
        image_id = image_file.split("_")[0]
        detections_val[image_id] = metrics_dict

        if save_preds:
            predictions_val[image_id] = {
                "boxes": boxes,
                "scores": boxes_conf,
                "labels": boxes_cls,
            }

    print(f"Mean IoU: {np.mean(iou_vals):.4f}")
    print(f"Mean LRP: {np.mean(lrp_vals):.4f}")

    if output_name != None:
        save_to_json(detections_val, output_dir, output_name)
    if save_preds:
        save_predictions(predictions_val, output_dir, "predictions.json")


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


def run_faster_rcnn(predictor: DefaultPredictor, dataset: list[dict], output_dir: str, output_name: str | None = None, discretize_threshold: float | None = None, save_zero_iou: bool = False, save_preds: bool = False) -> None:
    detections_val = {}
    predictions_val = {}
    iou_vals = []
    lrp_vals = []
    for d in tqdm(dataset):
        image_path = d["file_name"]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}.")
            continue

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
        annotations = annotations_from_coco(d)

        if detections[0]["boxes"].numel() == 0:
            metrics_dict = {"iou": 0.0, "lrp": 0.0, "confidence": []}
        elif annotations[0]["boxes"].numel() == 0:
            print(f"WARNING: No annotations found for image {image_path}.")
            metrics_dict = {"iou": None, "lrp": None, "confidence": []}
        else:
            metrics_dict = metrics.get_metrics(detections, annotations, metrics=["iou", "lrp"])
            metrics_dict["confidence"] = scores.tolist()

        if save_zero_iou and metrics_dict.get("iou") == 0.0:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_iou0_image(image, detections[0]["boxes"], annotations[0]["boxes"], metrics_dict["iou"], os.path.join(output_dir, "iou0_images"), image_name)

        if discretize_threshold != None:
            metrics_dict = discretize_metrics(metrics_dict, discretize_threshold)

        iou_val = metrics_dict.get("iou", 0.0)
        lrp_val = metrics_dict.get("lrp", 0.0)
        if iou_val is not None:
            iou_vals.append(iou_val)
            lrp_vals.append(lrp_val)
        image_id = str(d.get("image_id", os.path.basename(image_path)))
        detections_val[image_id] = metrics_dict

        if save_preds:
            predictions_val[image_id] = {
                "boxes": boxes.cpu().numpy().tolist(),
                "scores": scores.cpu().numpy().tolist(),
                "labels": labels.cpu().numpy().tolist(),
            }

    print(f"Mean IoU: {np.mean(iou_vals):.4f}")
    print(f"Mean LRP: {np.mean(lrp_vals):.4f}")

    if output_name != None:
        save_to_json(detections_val, output_dir, output_name)
    if save_preds:
        save_predictions(predictions_val, output_dir, "predictions.json")


def save_to_json(all_detections: dict, output_folder: str, filename: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    json_output_path = os.path.join(output_folder, filename)
    with open(json_output_path, "w") as json_file:
        json.dump(all_detections, json_file, indent=4)
    print(f"All detections saved to {json_output_path}")
    print(f"Total images processed: {len(all_detections)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["yolo", "rf-detr", "faster-rcnn"], type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--discretize-threshold", type=float, default=None)
    parser.add_argument("--save-zero-iou", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    run_train = args.train
    run_test = args.test

    output_name = "detections.json" if args.discretize_threshold == None else "detections_disc.json"

    # Yolo logic
    if args.model == "yolo":
        model_path = Path(args.checkpoint)
        model = YOLO(model_path)
        if run_train:
            print(f"Processing YOLO train images from ./data/zod_yolo/images/train")
            run_yolo(model, "./data/zod_yolo/images/train", "./results/yolo/", discretize_threshold=args.discretize_threshold, save_zero_iou=args.save_zero_iou, save_preds=True)
        if run_test:
            print(f"Processing YOLO test images from ./data/zod_yolo/images/test/")
            run_yolo(model,"./data/zod_yolo/images/test/","./results/yolo/",output_name,discretize_threshold=args.discretize_threshold, save_zero_iou=args.save_zero_iou, save_preds=True)

    # Faster RCNN logic
    elif args.model == "faster-rcnn":
        cfg = get_cfg()
        cfg.merge_from_file("./models/faster-rcnn/config.yaml")
        cfg.MODEL.WEIGHTS = args.checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.freeze()
        predictor = DefaultPredictor(cfg)
        if run_train:
            register_coco_instances("train", {}, os.path.join("./data/zod_coco/", "train", "_annotations.coco.json"), ".")
            train_dataset = DatasetCatalog.get("train")
            print(f"Processing Faster R-CNN train dataset ({len(train_dataset)} images)")
            run_faster_rcnn(
                predictor,
                train_dataset,
                "results/faster-rcnn",
                discretize_threshold=args.discretize_threshold, save_zero_iou=args.save_zero_iou, save_preds=True)
        if run_test:
            register_coco_instances("test", {}, os.path.join("./data/zod_coco/", "test", "_annotations.coco.json"), ".")
            test_dataset = DatasetCatalog.get("test")
            print(f"Processing Faster R-CNN test dataset ({len(test_dataset)} images)")
            run_faster_rcnn(predictor,test_dataset,"results/faster-rcnn",output_name,discretize_threshold=args.discretize_threshold, save_zero_iou=args.save_zero_iou, save_preds=True)

    # Rfdetr logic
    elif args.model == "rf-detr":
        model_path = Path(args.checkpoint)
        model = RFDETRBase(pretrain_weights=model_path)
        if run_train:
            print(f"Processing RF-DETR train images from {args.yolo_train}")
            run_rf_detr(model,"./data/zod_yolo/images/train/","results/rf-detr",discretize_threshold=args.discretize_threshold, save_zero_iou=args.save_zero_iou, save_preds=True)
        if run_test:
            print(f"Processing RF-DETR test images from {args.yolo_test}")
            run_rf_detr(model,"./data/zod_yolo/images/test/","results/rf-detr",output_name,discretize_threshold=args.discretize_threshold, save_zero_iou=args.save_zero_iou, save_preds=True)



if __name__ == "__main__":
    main()
