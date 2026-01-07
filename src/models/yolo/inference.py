from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import os
import json
import cv2
from src.models import metrics
import torch
from torch import tensor
import numpy as np


def load_model():
    #model_path = Path("./models/yolo11s.pt")
    model_path = Path("./models/yolo_experiments/train/weights/best.pt")

    if model_path.exists():
        print("Loading existing model")
        model = YOLO(model_path)
    else:
        raise NameError

    return model


def process_image(image_path: str, model, output_folder: str) -> dict:
    name = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return {"coordinates": [], "class": [], "confidence": []}
    
    img_height, img_width = image.shape[:2]
    annotations = load_yolo_annotations(annotation_path, img_width, img_height)

    results = model(image, verbose=False)[0]
    
    boxes = [box.xyxy.cpu().numpy()[0].tolist() for box in results.boxes]
    boxes_conf = [box.conf.cpu().numpy()[0].item() for box in results.boxes]
    boxes_cls = [box.cls.cpu().numpy()[0].item() for box in results.boxes]
    
    detections = [{
    "boxes": torch.tensor(boxes, dtype=torch.float32),
    "scores": torch.tensor(boxes_conf, dtype=torch.float32),
    "labels": torch.tensor(boxes_cls, dtype=torch.long)}]
    
    if detections[0]["boxes"].numel() == 0:
        #print(f"WARNING: No detections found for image {image_path}.")
        metrics_dict = {"iou": 0.0, "lrp": 0.0, "confidence": []}
        return metrics_dict
    elif annotations[0]["boxes"].numel() == 0:
        print(f"WARNING: No annotations found for image {image_path}.")
        metrics_dict = {"iou": None, "lrp": None, "confidence": []}
        return metrics_dict

    metrics_dict = metrics.get_metrics(detections, annotations, metrics=["iou", "lrp"])
    metrics_dict["confidence"] = boxes_conf
    iou = metrics_dict["iou"]
    for i, box in enumerate(boxes):
        x, y, x1, y1 = box
        conf = boxes_conf[i]
        cls = boxes_cls[i]
        cv2.rectangle(image, (int(x), int(y)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.putText(image, f"Pred box {cls}: {iou:.2f}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    for ann in annotations[0]["boxes"]:
        #ann_class = ann["labels"].item()
        x1, y1 = int(ann[0]), int(ann[1])
        x2, y2 = int(ann[2]), int(ann[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"GT box", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the processed image
    #output_path = os.path.join(output_folder, f'{name}_detected.jpg')
    output_path = os.path.join("results/yolo/iou_0", f'{name}_detected.jpg')
    if iou == 0.0:
        cv2.imwrite(output_path, image)
    #print(f'Output image saved as {output_path}')
    
    return metrics_dict



def save_to_json(all_detections: dict, output_folder: str) -> None:
    json_output_path = os.path.join(output_folder, 'detections_400e.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(all_detections, json_file, indent=4)
    
    print(f'All detections saved to {json_output_path}')
    print(f'Total images processed: {len(all_detections)}')


def load_yolo_annotations(annotation_path: str, img_width: int, img_height: int) -> list[dict]:
    boxes = []
    labels = []
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
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


RUN_TRAIN = True
RUN_VAL = True

if __name__ == "__main__":
    input_path_train = './data/zod_yolo/images/train/'
    input_path_val = './data/zod_yolo/images/val/'
    output_folder = './results/yolo/'
    output_folder_images = os.path.join(output_folder, "images")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_images, exist_ok=True)

    model = load_model()

    image_extensions = ['.jpg', '.png', '.jpeg']
    images_files_train = [f for f in os.listdir(input_path_train) if any(f.lower().endswith(ext) for ext in image_extensions)]
    images_files_val = [f for f in os.listdir(input_path_val) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if RUN_TRAIN:
        print(f"Processing {len(images_files_train)} training images")
        iou_train = []
        lrp_train = []
        map_train = []
        for image_file in tqdm(images_files_train):
            id = image_file.split("_")[0]
            image_path = os.path.join(input_path_train, image_file)
            results = process_image(image_path, model, output_folder_images)
            iou_val = results.get("iou", 0.0)
            lrp_val = results.get("lrp", 0.0)
            if iou_val is None:
                continue
            iou_train.append(iou_val)
            lrp_train.append(lrp_val)

        print(f"Training Mean IoU: {np.mean(iou_train):.4f}")
        print(f"Training Mean LRP: {np.mean(lrp_train):.4f}")
        
    if RUN_VAL:
        print(f"Processing {len(images_files_val)} validation images")
        detections_val = {}
        iou_vals = []
        lrp_vals = []
        for image_file in tqdm(images_files_val):
            id = image_file.split("_")[0]
            image_path = os.path.join(input_path_val, image_file)
            results = process_image(image_path, model, output_folder_images)
            iou_val = results.get("iou", 0.0)
            lrp_val = results.get("lrp", 0.0)
            if iou_val is None:
                continue
            iou_vals.append(iou_val)
            lrp_vals.append(lrp_val)
            detections_val[id] = results

        print(f"Validation Mean IoU: {np.mean(iou_vals):.4f}")
        print(f"Validation Mean LRP: {np.mean(lrp_vals):.4f}")
        
        save_to_json(detections_val, output_folder)
