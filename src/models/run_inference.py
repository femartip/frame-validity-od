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

IMG_SIZE = (1344, 768)

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


def save_iou0_image(image, pred, gt, iou, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    img = image.copy()
    pred_boxes = pred["boxes"]
    gt_boxes = gt["boxes"]
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy().tolist()
        pred_labels =  pred["labels"].cpu().numpy().tolist()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy().tolist()
        gt_labels = gt["labels"].cpu().numpy().tolist()
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.putText(img, f"pred, cls {pred_labels[i]}", (x1, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    for j, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.putText(img, f"gt, cls {gt_labels[j]}", (x1, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, f"IoU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_iou0.jpg"), img)


def save_predictions(predictions_dict, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, filename)
    with open(json_output_path, "w") as json_file:
        json.dump(predictions_dict, json_file, indent=4)
    print(f"Predictions saved to {json_output_path}")


def _open_raw_writer(output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, filename)
    return open(json_output_path, "w"), json_output_path


def _write_raw_prediction(handle, image_id: str, payload: dict) -> None:
    record = {"image_id": image_id, **payload}
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def _write_raw_metadata(output_dir: str, filename: str, metadata: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, filename)
    with open(meta_path, "w") as json_file:
        json.dump(metadata or {}, json_file, indent=4)
    print(f"Raw predictions metadata saved to {meta_path}")


def _infer_yolo_num_classes(model: YOLO) -> int:
    if hasattr(model, "model") and hasattr(model.model, "nc"):
        return int(model.model.nc)
    if hasattr(model, "names"):
        return int(len(model.names))
    return 0


def _get_yolo_nms_iou(model: YOLO) -> float:
    if hasattr(model, "overrides") and isinstance(model.overrides, dict):
        if "iou" in model.overrides:
            return float(model.overrides["iou"])
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        args = getattr(predictor, "args", None)
        if args is not None and hasattr(args, "iou"):
            return float(args.iou)
    return 0.7


def _extract_yolo_raw_predictions(model: YOLO, image: np.ndarray) -> tuple[list, list, list, list]:
    device = next(model.model.parameters()).device  # type: ignore
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).to(device).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    raw = model.model(tensor)  # type: ignore
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], torch.Tensor):
        if raw[0].ndim in (4, 5):
            # Detection heads output: flatten each head to (bs, -1, no) then concat.
            flat = []
            for r in raw:
                if r.ndim == 4:
                    # (bs, no, h, w)
                    bs, no, h, w = r.shape
                    r = r.view(bs, no, h * w).permute(0, 2, 1)
                elif r.ndim == 5:
                    # (bs, anchors, h, w, no)
                    bs, na, h, w, no = r.shape
                    r = r.view(bs, na * h * w, no)
                flat.append(r)
            raw = torch.cat(flat, dim=1)
        else:
            raw = raw[0]
    if isinstance(raw, dict) and "pred" in raw:
        raw = raw["pred"]
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim == 2 and raw.shape[0] < raw.shape[1] and raw.shape[0] <= 512:
        # Likely (no, num_preds) instead of (num_preds, no)
        raw = raw.T
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError(f"Unexpected YOLO raw shape: {tuple(raw.shape)}")

    boxes_xywh = raw[:, :4].clone()
    scores_raw = raw[:, 4:]
    if scores_raw.max() > 1.0 or scores_raw.min() < 0.0:
        scores_raw = scores_raw.sigmoid()
    obj = scores_raw[:, 0:1]
    class_probs = scores_raw[:, 1:]

    if boxes_xywh.max() <= 2.0:
        boxes_xywh[:, 0] *= image.shape[1]
        boxes_xywh[:, 2] *= image.shape[1]
        boxes_xywh[:, 1] *= image.shape[0]
        boxes_xywh[:, 3] *= image.shape[0]

    x = boxes_xywh[:, 0]
    y = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    scores = (class_probs * obj).max(dim=1).values
    classes = (class_probs * obj).argmax(dim=1)
    return (
        boxes.detach().cpu().numpy().tolist(),
        scores.detach().cpu().numpy().tolist(),
        class_probs.detach().cpu().numpy().tolist(),
        classes.detach().cpu().numpy().tolist(),
    )


def _extract_faster_rcnn_raw_predictions(predictor: DefaultPredictor, image: np.ndarray) -> tuple[list, list, list, list]:
    model = predictor.model
    with torch.no_grad():
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)
        features_list = [features[f] for f in model.roi_heads.in_features]
        box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(box_features)
        proposal_boxes = proposals[0].proposal_boxes.tensor
        boxes = model.roi_heads.box_predictor.box2box_transform.apply_deltas(
            pred_proposal_deltas, proposal_boxes
        )
        probs = torch.softmax(pred_class_logits, dim=1)
        class_probs = probs[:, :-1]
        scores, classes = class_probs.max(dim=1)
        keep = scores > 0
        boxes = boxes[keep]
        class_probs = class_probs[keep]
        scores = scores[keep]
        classes = classes[keep]

        if boxes.ndim == 2 and boxes.shape[1] > 4:
            num_classes = class_probs.shape[1]
            boxes = boxes.view(-1, num_classes, 4)
            idx = torch.arange(boxes.shape[0], device=boxes.device)
            boxes = boxes[idx, classes]

        return (
            boxes.cpu().numpy().tolist(),
            scores.cpu().numpy().tolist(),
            class_probs.cpu().numpy().tolist(),
            classes.cpu().numpy().tolist(),
        )


def run_yolo(
    model: YOLO,
    image_root: str,
    output_dir: str,
    output_name: str | None = None,
    discretize_threshold: float | None = None,
    save_zero_iou: bool = False,
    save_preds: bool = False,
    save_raw: bool = False,
) -> None:
    detections_val = {}
    predictions_val = {}
    raw_writer = None
    raw_path = None
    nms_iou = _get_yolo_nms_iou(model)
    num_classes = _infer_yolo_num_classes(model)
    iou_vals = []
    lrp_vals = []
    image_files = list_images(image_root)
    if save_raw:
        raw_writer, raw_path = _open_raw_writer(output_dir, "raw_predictions.jsonl")

    for image_file in tqdm(image_files):
        image_path = os.path.join(image_root, image_file)

        #Process image logic
        annotation_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
        image = cv2.imread(image_path)
        image = cv2.resize(image, IMG_SIZE)
        
        img_height, img_width = image.shape[:2]
        annotations = load_yolo_annotations(annotation_path, img_width, img_height)

        if save_raw:
            raw_boxes, raw_scores, raw_probs, raw_classes = _extract_yolo_raw_predictions(model, image)
            _write_raw_prediction(
                raw_writer,
                image_file.split("_")[0],
                {
                "boxes": raw_boxes,
                "scores": raw_scores,
                "class_probs": raw_probs,
                "classes": raw_classes,
                },
            )

        results = model.predict(image, imgsz=IMG_SIZE, verbose=False)[0]
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
            #metrics_dict = {"iou": None, "lrp": None, "confidence": []}
            continue
        else:
            metrics_dict = metrics.get_metrics(detections, annotations, metrics=["iou", "lrp"])
            metrics_dict["confidence"] = boxes_conf

        if save_zero_iou and metrics_dict.get("iou") == 0.0:
            image_name = os.path.splitext(image_file)[0]
            save_iou0_image(image, detections[0], annotations[0], metrics_dict["iou"], os.path.join(output_dir, "iou0_images"), image_name)

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
    if save_raw:
        if raw_writer:
            raw_writer.close()
        if num_classes == 0 and raw_path:
            try:
                with open(raw_path, "r") as f:
                    first = json.loads(f.readline())
                    num_classes = len(first.get("class_probs", [0]))
            except Exception:
                pass
        _write_raw_metadata(
            output_dir,
            "raw_predictions.meta.json",
            {"nms_iou": nms_iou, "num_classes": num_classes, "model": "yolo"},
        )


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
        image = cv2.resize(image, IMG_SIZE)

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
            save_iou0_image(image, detections[0], annotations[0], metrics_dict["iou"], os.path.join(output_dir, "iou0_images"), image_name)

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
    orig_w = dataset_dict.get("width", None)
    orig_h = dataset_dict.get("height", None)

    if orig_w is None or orig_h is None:
        # fallback: read from file (slower)
        img0 = cv2.imread(dataset_dict["file_name"])
        orig_h, orig_w = img0.shape[:2]

    sx = IMG_SIZE[0] / orig_w
    sy = IMG_SIZE[1] / orig_h

    boxes, labels = [], []
    for ann in dataset_dict.get("annotations", []):
        x, y, w, h = ann["bbox"]  # COCO xywh in original pixels
        x1 = x * sx
        y1 = y * sy
        x2 = (x + w) * sx
        y2 = (y + h) * sy
        boxes.append([x1, y1, x2, y2])
        labels.append(int(ann["category_id"]))

    return [{
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }]


def run_faster_rcnn(
    predictor: DefaultPredictor,
    dataset: list[dict],
    output_dir: str,
    output_name: str | None = None,
    discretize_threshold: float | None = None,
    save_zero_iou: bool = False,
    save_preds: bool = False,
    save_raw: bool = False,
) -> None:
    detections_val = {}
    predictions_val = {}
    raw_writer = None
    raw_path = None
    nms_iou = float(predictor.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
    num_classes = int(predictor.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    iou_vals = []
    lrp_vals = []
    if save_raw:
        raw_writer, raw_path = _open_raw_writer(output_dir, "raw_predictions.jsonl")
    for d in tqdm(dataset):
        image_path = d["file_name"]
        image = cv2.imread(image_path)
        image = cv2.resize(image, IMG_SIZE)

        if image is None:
            print(f"Error: Could not read image {image_path}.")
            continue

        if save_raw:
            raw_boxes, raw_scores, raw_probs, raw_classes = _extract_faster_rcnn_raw_predictions(predictor, image)
            image_id = str(d.get("image_id", os.path.basename(image_path)))
            _write_raw_prediction(
                raw_writer,
                image_id,
                {
                    "boxes": raw_boxes,
                    "scores": raw_scores,
                    "class_probs": raw_probs,
                    "classes": raw_classes,
                },
            )

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
            save_iou0_image(image, detections[0], annotations[0], metrics_dict["iou"], os.path.join(output_dir, "iou0_images"), image_name)

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
    if save_raw:
        if raw_writer:
            raw_writer.close()
        if num_classes == 0 and raw_path:
            try:
                with open(raw_path, "r") as f:
                    first = json.loads(f.readline())
                    num_classes = len(first.get("class_probs", [0]))
            except Exception:
                pass
        _write_raw_metadata(
            output_dir,
            "raw_predictions.meta.json",
            {"nms_iou": nms_iou, "num_classes": num_classes, "model": "faster-rcnn"},
        )


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
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--discretize-threshold", type=float, default=None)
    parser.add_argument("--save-zero-iou", action="store_true")
    parser.add_argument("--save-raw", action="store_true", help="Save pre-NMS raw predictions for MetaDetect features")
    return parser.parse_args()

def main():
    args = parse_args()
    run_train = args.train
    run_validate = args.validate    
    run_test = args.test

    output_name = "detections.json" if args.discretize_threshold == None else "detections_disc.json"

    # Yolo logic
    if args.model == "yolo":
        model_path = Path(args.checkpoint)
        model = YOLO(model_path)
        if run_train:
            print(f"Processing YOLO train images from ./data/zod_yolo/images/train")
            run_yolo(
                model,
                "./data/zod_yolo/images/train",
                "./results/yolo/",
                discretize_threshold=args.discretize_threshold,
                save_raw=args.save_raw,
            )
        if run_validate:
            print(f"Processing YOLO validation images from ./data/zod_yolo/images/val/")
            run_yolo(
                model,
                "./data/zod_yolo/images/val/",
                "./results/yolo/",
                output_name,
                discretize_threshold=args.discretize_threshold,
                save_raw=args.save_raw,
            )
        if run_test:
            print(f"Processing YOLO test images from ./data/zod_yolo/images/test/")
            run_yolo(
                model,
                "./data/zod_yolo/images/test/",
                "./results/yolo/",
                output_name,
                discretize_threshold=args.discretize_threshold,
                save_zero_iou=args.save_zero_iou,
                save_preds=True,
                save_raw=args.save_raw,
            )

    # Faster RCNN logic
    elif args.model == "faster-rcnn":
        cfg = get_cfg()
        cfg.set_new_allowed(True)  # allow custom keys like DATASETS.VAL saved in training config
        cfg.merge_from_file("./models/faster-rcnn/config.yaml")
        cfg.set_new_allowed(False)
        cfg.MODEL.WEIGHTS = args.checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.freeze()
        predictor = DefaultPredictor(cfg)
        if run_train:
            #register_coco_instances("train", {}, os.path.join("./data/zod_coco/", "train", "_annotations.coco.json"), ".")
            register_coco_instances("train", {}, os.path.join("./data/zod_coco/", "zod_full_Anonymization.BLUR_train.json"), ".")
            train_dataset = DatasetCatalog.get("train")
            print(f"Processing Faster R-CNN train dataset ({len(train_dataset)} images)")
            run_faster_rcnn(
                predictor,
                train_dataset,
                "results/faster-rcnn",
                discretize_threshold=args.discretize_threshold,
                save_raw=args.save_raw,
            )
        if run_validate:
            #register_coco_instances("valid", {}, os.path.join("./data/zod_coco/", "valid", "_annotations.coco.json"), ".")
            register_coco_instances("valid", {}, os.path.join("./data/zod_coco/", "zod_full_Anonymization.BLUR_val.json"), ".")
            val_dataset = DatasetCatalog.get("valid")
            print(f"Processing Faster R-CNN validation dataset ({len(val_dataset)} images)")
            run_faster_rcnn(
                predictor,
                val_dataset,
                "results/faster-rcnn",
                output_name,
                discretize_threshold=args.discretize_threshold,
                save_raw=args.save_raw,
            )
        if run_test:
            #register_coco_instances("test", {}, os.path.join("./data/zod_coco/", "test", "_annotations.coco.json"), ".")
            register_coco_instances("test", {}, os.path.join("./data/zod_coco/", "zod_full_Anonymization.BLUR_test.json"), ".")
            test_dataset = DatasetCatalog.get("test")
            print(f"Processing Faster R-CNN test dataset ({len(test_dataset)} images)")
            run_faster_rcnn(
                predictor,
                test_dataset,
                "results/faster-rcnn",
                output_name,
                discretize_threshold=args.discretize_threshold,
                save_zero_iou=args.save_zero_iou,
                save_preds=True,
                save_raw=args.save_raw,
            )

    # Rfdetr logic
    elif args.model == "rf-detr":
        model_path = Path(args.checkpoint)
        model = RFDETRBase(pretrain_weights=model_path)
        if run_train:
            print(f"Processing RF-DETR train images from {args.yolo_train}")
            run_rf_detr(model,"./data/zod_yolo/images/train/","results/rf-detr",discretize_threshold=args.discretize_threshold)
        if run_validate:
            print(f"Processing RF-DETR validation images from {args.yolo_valid}")
            run_rf_detr(model,"./data/zod_yolo/images/val/","results/rf-detr",output_name,discretize_threshold=args.discretize_threshold)
        if run_test:
            print(f"Processing RF-DETR test images from {args.yolo_test}")
            run_rf_detr(model,"./data/zod_yolo/images/test/","results/rf-detr",output_name,discretize_threshold=args.discretize_threshold, save_zero_iou=args.save_zero_iou, save_preds=True)



if __name__ == "__main__":
    main()
    
