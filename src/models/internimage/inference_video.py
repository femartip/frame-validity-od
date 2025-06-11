import cv2
import os
from tqdm import tqdm
import argparse
import csv
import json
import numpy as np

import torch
import mmcv
from mmdet.apis import init_detector, inference_detector

from classification.functons_cls import initialize_model as initialize_model_cls
from classification.functons_cls import classification_oneDet
from colorama import Fore, Style

def process_detections(det_results, thr_det):
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


def process_imgs(image, model_det, score_thr, model_cls, cls_names, score_cls):    
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
    

def getframes (video_path, output_folder, model_det, score_thr, model_cls, cls_names, score_cls):
    name = video_path.split('/')[-1].split('.')[0]
    
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size_img = (width, height)

    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{output_folder}{name}.avi', codec, fps, size_img)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        cls_detec_list = process_imgs(frame, model_det, score_thr, model_cls, cls_names, score_cls)
        # print(len(cls_detec_list), 'detections in frame', frame_count)
        # Draw the information
        for info in cls_detec_list:
            x, y, w, h = info['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{info['class']}: {info['score']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
    out.release()
    print(Fore.GREEN + f'Output video saved in {name}' + Fore.RESET)
    cap.release()


def argsparse():
    parser = argparse.ArgumentParser(description="Process images to detect traffic signs.")
    parser.add_argument('--cfg_det', default='detection/configs/zenseact/dino_4scale_internimage_l_3x_zenseact_0.1x_backbone_lr.py', help='Config file')
    parser.add_argument('--ckpt_det', default='/data1/weights/TrafficSign/Detection/dcnv3/oneClass_TrafficSign/best_bbox_mAP_epoch_14.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:1', help='Device used for inference')
    parser.add_argument('--score_thr', type=float, default=0.65, help='Bbox score threshold')
        
    parser.add_argument('--cfg_cls', default='/data1/training_trafficSign/Weights/classification/testing/zenseact_dataset13_04_2025/internimage_s_1k_224/config.yaml', help='Config file')
    parser.add_argument('--ckpt_cls', default='/data1/training_trafficSign/Weights/classification/testing/zenseact_dataset13_04_2025/internimage_s_1k_224/ckpt_epoch_ema_best.pth', help='Checkpoint file')
    parser.add_argument('--cls_names', default='/data1/training_trafficSign/Weights/classification/testing/zenseact_dataset13_04_2025/id2class.npy', help='Npy file with the names of the class for cls model')
    parser.add_argument('--score_cls', type=float, default=0.64, help='Bbox score threshold')
    return parser.parse_args()


if __name__=='__main__':    
    # Load input video
    input_path = '/data1/capsules/trafficSign/sw400/DEV_HCB_2005_AEB_20221210_094307/'
    output_frames = '/data1/users/'
    
    args = argsparse()
    
    # inizilaize the models
    torch.cuda.set_device(args.device)
    model_det = init_detector(args.cfg_det, args.ckpt_det, device=args.device)
    
    model_cls = initialize_model_cls(args.cfg_cls, args.ckpt_cls)
    cls_names = np.load(args.cls_names, allow_pickle='TRUE').item()
    
    for video in tqdm(os.listdir(input_path)):
        
        if video.endswith('.mp4') or video.endswith('.avi'):
            getframes(f'{input_path}/{video}', output_frames, model_det, args.score_thr, model_cls, cls_names, args.score_cls)