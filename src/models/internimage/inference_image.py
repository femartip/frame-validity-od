import cv2
import os
from tqdm import tqdm
import argparse
import csv
import json
import numpy as np

import torch
from torch import nn
import mmcv
from mmdet.apis import init_detector, inference_detector

from functions_cls import initialize_model as initialize_model_cls
from functions_cls import classification_oneDet
from colorama import Fore, Style

def process_detections(det_results: list, thr_det: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes = np.vstack(det_results)
    labels = np.concatenate([
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(det_results)
    ])

    assert bboxes.shape[1] == 5
    scores = bboxes[:, -1]
    inds = scores > thr_det
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    scores = scores[inds]

    return bboxes, scores, labels


def process_imgs(image: cv2.Mat, model_det: nn.Module, score_thr: float, model_cls, cls_names: np.ndarray, score_cls: float) -> list[dict]:    
    # information of static objects 
    cls_detec_list = []
    
    # Perform traffic sign detection on the image
    result = inference_detector(model_det, image)
    boxes_det, scores_det, labels_det = process_detections(result, score_thr)
    
    hgt_img, wdt_img, _ = image.shape
    bboxes_det = []
    for bbox, _, _ in zip(boxes_det, scores_det, labels_det):
        bbox_int = bbox.astype(np.int32)
        width = int(bbox_int[2] - bbox_int[0])
        height = int(bbox_int[3] - bbox_int[1])
        rect = [int(bbox_int[0]), int(bbox_int[1]), int(width), int(height)]
        bboxes_det.append(rect)
        

    # Create a JSON file for each bounding box
    for i, box in enumerate(bboxes_det):
        x, y, w, h = box
        
        # Extract ROI and resize for classification
        roi = image[y:y+h, x:x+w]
        roi_resize = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_CUBIC)

        # Perform classification on the ROI
        class_out, score_h = classification_oneDet(model_cls, roi_resize)

        # Determine class name and score
        if score_h >= score_cls:
            name_cls = cls_names[class_out]
           
        else:
            name_cls = "NotListed"
            score_h = 0.0
            
        aux_info = {'class': name_cls,
                    'score': score_h,
                    'bbox': [x, y, w, h]}
        
        # print(aux_info)
        cls_detec_list.append(aux_info)
    return cls_detec_list
    

def process_image(image_path: str, output_folder: str, model_det: nn.Module, score_thr: float, model_cls, cls_names: np.ndarray, score_cls: float) -> list[dict]:
    name = os.path.splitext(os.path.basename(image_path))[0]
    
    os.makedirs(output_folder, exist_ok=True)

    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return []
    
    cls_detec_list = process_imgs(image, model_det, score_thr, model_cls, cls_names, score_cls)
    print(f'{len(cls_detec_list)} detections in image {name}')
    
    # Draw the information
    for info in cls_detec_list:
        x, y, w, h = info['bbox']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{info['class']}: {info['score']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save the processed image
    output_path = os.path.join(output_folder, f'{name}_detected.jpg')
    cv2.imwrite(output_path, image)
    print(Fore.GREEN + f'Output image saved as {output_path}' + Fore.RESET)
    
    return cls_detec_list

def save_to_json(all_detections: dict) -> None:
    json_output_path = os.path.join(output_folder, 'detections.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(all_detections, json_file, indent=4)
    
    print(Fore.CYAN + f'All detections saved to {json_output_path}' + Fore.RESET)
    print(f'Total images processed: {len(all_detections)}')

def argsparse():
    parser = argparse.ArgumentParser(description="Process images to detect traffic signs.")
    parser.add_argument('--cfg_det', default='./models/internimage_detection/dcnv4/dino_4scale_flash_internimage_b_1x_zenseact.py', help='Config file')
    parser.add_argument('--ckpt_det', default='./models/internimage_detection/dcnv4/best_bbox_mAP_epoch_12.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score_thr', type=float, default=0.65, help='Bbox score threshold')
        
    parser.add_argument('--cfg_cls', default='./models/internimage_classification/training__dataset21_01_2025/config.yaml', help='Config file')
    parser.add_argument('--ckpt_cls', default='./models/internimage_classification/training__dataset21_01_2025/ckpt_epoch_ema_best.pth', help='Checkpoint file')
    parser.add_argument('--cls_names', default='./id2class.npy', help='Npy file with the names of the class for cls model')
    parser.add_argument('--score_cls', type=float, default=0.64, help='Bbox score threshold')
    return parser.parse_args()


if __name__=='__main__':    
    # Load input images
    input_path = './zod_yolo/images/val/'
    output_folder = './results/internimage/'

    args = argsparse()
    
    # inizilaize the models
    torch.cuda.set_device(args.device)
    model_det = init_detector(args.cfg_det, args.ckpt_det, device=args.device)
    
    model_cls = initialize_model_cls(args.cfg_cls, args.ckpt_cls)
    cls_names = np.load(args.cls_names, allow_pickle=True).item()
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(input_path) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    all_detections = {}
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_path, image_file)
        image_detection = process_image(image_path, output_folder, model_det, args.score_thr, model_cls, cls_names, args.score_cls)
        all_detections[image_file] = image_detection

    save_to_json(all_detections)