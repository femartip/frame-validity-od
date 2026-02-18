from ultralytics import YOLO  #type: ignore
from pathlib import Path
import tensorboard

import argparse
from dotenv import load_dotenv
import os
from huggingface_hub import HfApi, create_repo

load_dotenv()

def upload_to_hub(model_dir: str, repo_name: str, token: str) -> None:
    api = HfApi()
    create_repo(repo_name, token=token, repo_type="model", exist_ok=True)
    #api.upload_folder(repo_id=repo_name, folder_path=model_dir, repo_type="model", token=token, path_in_repo=".", commit_message="Upload Detectron2 Faster R-CNN checkpoint + config")
    
    api.upload_file(path_or_fileobj=f"{model_dir}/args.yaml", path_in_repo="args.yaml", repo_id=repo_name, repo_type="model")
    api.upload_file(path_or_fileobj=f"{model_dir}/MODEL_CARD.md", path_in_repo="README.md", repo_id=repo_name, repo_type="model")
    # Upload all tensorboard event files
    for filename in os.listdir(model_dir):
        if filename.startswith("events.out.tfevents"):
            api.upload_file(path_or_fileobj=os.path.join(model_dir, filename), path_in_repo=filename, repo_id=repo_name, repo_type="model")
    for filename in os.listdir(os.path.join(model_dir, "weights")):
        if filename.endswith(".pt"):
            api.upload_file(path_or_fileobj=os.path.join(model_dir, "weights", filename), path_in_repo=filename, repo_id=repo_name, repo_type="model")


def load_model(resume: bool) -> YOLO:
    if resume:
        model_path = Path("./models/yolo/train/weights/best.pt")
    else:
        model_path = "./models/yolo11l.pt"

    if Path(model_path).exists():
        print("Loading existing model")
        model = YOLO(model_path)
    else:
        print("Model does not exist, will load new model")
        model = YOLO("yolo11l.pt")
    return model

def train_yolo(model: YOLO, resume: bool)-> dict:
    train_results = model.train(
        data="./data/zod_yolo/dataset.yaml",  # Path to dataset configuration file
        project="./models/yolo/",
        save_dir="./models/yolo/train",
        name="train",
        workers=16,
        amp=False,
        save=True,
        save_period=10,
        resume=resume,   # Resuming training
        imgsz=512,
        rect=True,
        pretrained=True,
        seed=43,
        device=0,
        patience=20,
        epochs=200,  # Number of training epochs
        batch=32,
        plots=True,
        exist_ok=True,
        lr0=0.1,
        multi_scale=0.25,
    )

    print("Training results:")
    print(train_results)
    return train_results

def eval_yolo(model: YOLO) -> dict:
    # Evaluate the model's performance on the validation set
    metrics = model.val(
        data="./data/zod_yolo/dataset.yaml",imgsz=512,batch=32,device=0)

    print("Evaluation results:")
    print(metrics)
    return metrics

def test_yolo(model: YOLO) -> dict:
    # Test the model's performance on the test set
    metrics = model.val(
        data="./data/zod_yolo/dataset.yaml",imgsz=512,batch=32,device=0,split="test")

    print("Test results:")
    print(metrics)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    
    model = load_model(args.resume)
    if args.eval_only:
        eval_yolo(model)
    elif args.test_only:
        test_yolo(model)
    elif args.push_to_hub:
        upload_to_hub("./models/yolo/train", "femartip/yolo-zod", os.environ["HF_TOKEN"])
    else:
        train_yolo(model, args.resume)
    