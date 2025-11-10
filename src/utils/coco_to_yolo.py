import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from ultralytics.utils import DATASETS_DIR, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.downloads import download, zip_directory
from ultralytics.utils.files import increment_path

def to_yaml(save_dir: str, dataset_info: dict) -> None:
    yaml_content = f"""# Dataset configuration
path: {save_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
"""
    for i, name in dataset_info['names'].items():
        yaml_content += f"  {i}: {name}\n"
    
    yaml_content += f"\nnc: {dataset_info['nc']}"
    
    with open(f"{save_dir}/dataset.yaml", "w") as f:
        f.write(yaml_content)




def convert_coco(labels_dir: str,save_dir: str,use_keypoints: bool = False, copy_images: bool = True) -> None:
    save_dir = Path(save_dir)
    
    dataset_info = {'names': {}, 'nc': 0}
    
    # Import json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = json_file.stem.replace("instances_", "")
        split = 'train' if 'train' in lname else ('val' if 'val' in lname else 'test')
        
        images_dir = save_dir / 'images' / split
        labels_dir_out = save_dir / 'labels' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir_out.mkdir(parents=True, exist_ok=True)

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        categories = {cat['id']: cat['name'] for cat in data['categories']}
        if not dataset_info['names']:  
            dataset_info['names'] = {i: categories[cat_id] for i, cat_id in enumerate(sorted(categories.keys()))}
            dataset_info['nc'] = len(categories)

        # Create image dict
        images = {x['id']: x for x in data["images"]}
        # Create image-annotations dict
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
            if img_id not in images:
                print(f"Missing ID {img_id}")
                continue
            
            img = images[img_id]
            h, w = img["height"], img["width"]
            f = img["file_name"]
            
            img_name = Path(f).name
            label_path = labels_dir_out / Path(img_name).with_suffix('.txt')
            label_path.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue

                cat_id = ann["category_id"]
                cls = list(categories.keys()).index(cat_id)

                if use_keypoints and "keypoints" in ann and ann["keypoints"]:
                    print("WARNING: Keypoints not bbox")                
                else:
                    bbox = np.array(ann["bbox"], dtype=np.float64)
                    bbox[:2] += bbox[2:] / 2  
                    bbox[[0, 2]] /= w  
                    bbox[[1, 3]] /= h  
                    if bbox[2] > 0 and bbox[3] > 0:
                        line = f"{cls} " + " ".join([f"{x:.6f}" for x in bbox])
                        lines.append(line)

            # Write
            with open(label_path, "w", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n" if lines else "")
            
            if copy_images:
                src_img_path = Path(f)  # Adjust path as needed
                dst_img_path = Path(f"{save_dir}/images/{split}/{img_name}")
                
                #print(src_img_path)
                if src_img_path.exists():
                    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        img = Image.open(src_img_path)
                        img = img.resize((1024, 1024))
                        img.save(dst_img_path)
                    except Exception as e:
                        print(f"Error processing image {src_img_path}: {e}")


    to_yaml(save_dir, dataset_info)
    print(f"Conversion complete! Dataset saved to {save_dir}")
    print(f"Classes: {dataset_info['nc']}, Names: {list(dataset_info['names'].values())}")

if __name__ == '__main__':
    convert_coco(
        labels_dir="./data/zod_coco/",
        save_dir="./data/zod_yolo/",
        #use_keypoints=True  # Since you're using keypoints data
        copy_images=True,
    )