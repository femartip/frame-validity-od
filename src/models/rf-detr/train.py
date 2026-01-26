from html import parser
from rfdetr import RFDETRBase, RFDETRSmall, RFDETRLarge # type: ignore
from pathlib import Path
import tensorboard

import argparse
from zod.anno.object import OBJECT_CLASSES
from dotenv import load_dotenv
import os
from huggingface_hub import HfApi, create_repo

load_dotenv()

def upload_to_hub(model_dir: str, repo_name: str, token: str) -> None:
    api = HfApi()
    create_repo(repo_name, token=token, repo_type="model", exist_ok=True)
    #api.upload_folder(repo_id=repo_name, folder_path=model_dir, repo_type="model", token=token, path_in_repo=".", commit_message="Upload Detectron2 Faster R-CNN checkpoint + config")
    
    api.upload_file(path_or_fileobj=f"{model_dir}/checkpoint_best_total.pth", path_in_repo="model_final.pth", repo_id=repo_name, repo_type="model")
    #api.upload_file(path_or_fileobj=f"{model_dir}/config.yaml", path_in_repo="config.yaml", repo_id=repo_name, repo_type="model")
    api.upload_file(path_or_fileobj=f"{model_dir}/MODEL_CARD.md", path_in_repo="README.md",      repo_id=repo_name, repo_type="model")
    # Upload all tensorboard event files
    for filename in os.listdir(model_dir):
        if filename.startswith("events.out.tfevents"):
            api.upload_file(path_or_fileobj=os.path.join(model_dir, filename), path_in_repo=filename, repo_id=repo_name, repo_type="model")
    



def train_rfdetr(model):
    train_results = model.train(
        dataset_dir="./data/zod_coco/",
        image_root=".",
        epochs=30,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-3,
        output_dir="./models/rf-detr/",
        early_stopping=True,
        early_stopping_patience=5,
        tensorboard=True,
        amp=True,
        num_workers=8,
        multi_scale=False,
        expanded_scales=False,
        use_ema=False,
        run_test=False,
        resolution=512,
        #resume=True
    )

    print("Training results:")
    print(train_results)
    return train_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    model = RFDETRLarge()
    train_rfdetr(model)
    if args.push_to_hub:
        upload_to_hub("./models/rf-detr/", "femartip/rf-detr-zod", os.environ["HF_TOKEN"])