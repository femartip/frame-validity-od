from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import os
import json
import cv2

def load_model():
    #model_path = Path("./models/yolo11s.pt")
    model_path = Path("./models/yolo_experiments/train6/weights/best.pt")

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
    
    boxes = [box.xywh.cpu().numpy()[0].tolist() for box in results.boxes]
    boxes_conf = [box.conf.cpu().numpy()[0].item() for box in results.boxes]
    boxes_cls = [box.cls.cpu().numpy()[0].item() for box in results.boxes]
    
    detections = {"coordinates": boxes, "class": boxes_cls, "confidence": boxes_conf}
    
    matches = match_detections_to_annotations(detections, annotations)

    iou_scores = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        conf = boxes_conf[i]
        cls = boxes_cls[i]
        match = next((m for m in matches if m['detection_idx'] == i), None)
        iou_scores.append(match['iou'] if match else 0.0)
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        cv2.putText(image, f"{cls}: {conf:.2f}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    for ann_class, ann_x, ann_y, ann_w, ann_h in annotations:
        x1, y1 = int(ann_x - ann_w/2), int(ann_y - ann_h/2)
        x2, y2 = int(ann_x + ann_w/2), int(ann_y + ann_h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    detections["iou"] = iou_scores
    # Save the processed image
    output_path = os.path.join(output_folder, f'{name}_detected.jpg')
    cv2.imwrite(output_path, image)
    #print(f'Output image saved as {output_path}')
    
    return detections



def save_to_json(all_detections: dict, output_folder: str) -> None:
    json_output_path = os.path.join(output_folder, 'detections.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(all_detections, json_file, indent=4)
    
    print(f'All detections saved to {json_output_path}')
    print(f'Total images processed: {len(all_detections)}')


def load_yolo_annotations(annotation_path: str, img_width: int, img_height: int) -> list:
    annotations = []
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    annotations.append([class_id, x_center, y_center, width, height])
    return annotations

def calculate_iou(box1: list[int], box2: list[int]) -> float:
    def center_to_corner(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    box1_corner = center_to_corner(box1)
    box2_corner = center_to_corner(box2)
    
    x1 = max(box1_corner[0], box2_corner[0])
    y1 = max(box1_corner[1], box2_corner[1])
    x2 = min(box1_corner[2], box2_corner[2])
    y2 = min(box1_corner[3], box2_corner[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    iou = ((x2 - x1) * (y2 - y1))/((box1[2] * box1[3]) + (box2[2] * box2[3]) - ((x2 - x1) * (y2 - y1)))
    return iou

def match_detections_to_annotations(detections, annotations: list, iou_threshold=0.5):
    matches = []
    used_annotations = set()
    
    for i, (det_box, det_class, det_conf) in enumerate(zip(detections['coordinates'], detections['class'], detections['confidence'])):
        best_iou = 0
        best_match = -1
        
        for j, (ann_class, ann_x, ann_y, ann_w, ann_h) in enumerate(annotations):
            if j in used_annotations:
                continue
            if int(det_class) == ann_class:
                iou = calculate_iou(det_box, [ann_x, ann_y, ann_w, ann_h])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = j
        
        if best_match != -1:
            used_annotations.add(best_match)
            matches.append({'detection_idx': i,'iou': best_iou})
        else:
            matches.append({'detection_idx': i, 'iou': 0.0})
    
    return matches



if __name__ == "__main__":
    input_path = './data/zod_yolo/images/val/'
    output_folder = './results/yolo/'
    output_folder_images = os.path.join(output_folder, "images")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_images, exist_ok=True)

    model = load_model()

    image_extensions = ['.jpg', '.png', '.jpeg']
    images_files = [f for f in os.listdir(input_path) if any(f.lower().endswith(ext) for ext in image_extensions)]
    #images_files = images_files[:1000]
    print(f"Processing {len(images_files)} images")

    all_detections = {}
    for image_file in tqdm(images_files):
        id = image_file.split("_")[0]
        image_path = os.path.join(input_path, image_file)
        image_detection = process_image(image_path, model, output_folder_images)
        all_detections[id] = image_detection

    save_to_json(all_detections, output_folder)