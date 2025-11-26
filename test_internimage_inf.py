import os
import sys
from tqdm import tqdm
import argparse

# Ensure custom ops package under InternImage/classification is discoverable.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "InternImage", "classification"))
sys.path.insert(0, os.path.join(BASE_DIR, "InternImage", "detection"))
import mmdet_custom  # registers custom DINO detector and related modules

from src.models.internimage.functions_cls import classification_oneDet
from src.models.internimage.functions_cls import initialize_model as initialize_model_cls
from colorama import Fore, Style

from mmdet.datasets.pipelines import Compose
from mmdet.apis import init_detector, inference_detector
from src.models.internimage.inference_image import process_image, save_to_json, argparse

def argsparse():
    parser = argparse.ArgumentParser(description="Process images to detect traffic signs.")
    parser.add_argument('--cfg_det', default='./models/internimage_detection/dcnv4/dino_4scale_flash_internimage_b_1x_zenseact.py', help='Config file')
    parser.add_argument('--ckpt_det', default='./models/internimage_detection/dcnv4/best_bbox_mAP_epoch_12.pth', help='Checkpoint file')
    #parser.add_argument('--cfg_det', default='./models/internimage_detection/dcnv3/dino_4scale_internimage_l_3x_zenseact_0.1x_backbone_lr.py', help='Config file')
    #parser.add_argument('--ckpt_det', default='./models/internimage_detection/dcnv3/best_bbox_mAP_epoch_14.pth', help='Checkpoint file')
    
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score_thr', type=float, default=0.0, help='Bbox score threshold')
        
    parser.add_argument('--cfg_cls', default='./models/internimage_classification/training__dataset21_01_2025/config.yaml', help='Config file')
    parser.add_argument('--ckpt_cls', default='./models/internimage_classification/training__dataset21_01_2025/ckpt_epoch_ema_best.pth', help='Checkpoint file')
    #parser.add_argument('--cls_names', default='./id2class.npy', help='Npy file with the names of the class for cls model')
    parser.add_argument('--score_cls', type=float, default=0.0, help='Bbox score threshold')
    return parser.parse_args()

if __name__=='__main__':    
    # Load input images
    input_path = '/home/felix/Predictability-AD/data/zod_yolo/images/val/'
    output_folder = './results/internimage/images/'
    args = argsparse()
    
    model_det = init_detector(args.cfg_det, args.ckpt_det, device=str(args.device))

    model_cls = initialize_model_cls(args.cfg_cls, args.ckpt_cls)
    #cls_names = np.load(args.cls_names, allow_pickle=True).item()

    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(input_path) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    all_detections = {}
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_path, image_file)
        image_detection = process_image(image_path, output_folder, model_det, args.score_thr, model_cls, args.score_cls)
        all_detections[image_file] = image_detection

    save_to_json(all_detections)
