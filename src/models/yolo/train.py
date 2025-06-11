from ultralytics import YOLO
from pathlib import Path
import tensorboard

def train_yolo():
    model_path = Path("./models/yolo11s.pt")

    if model_path.exists():
        model = YOLO(model_path)
    else:
        model = YOLO("yolo11s.pt")

    train_results = model.train(
        data="./data/zod_yolo/dataset.yaml",  # Path to dataset configuration file
        project="./models/yolo_experiments/",
        imgsz=1024,
        pretrained=True,
        device=1,
        epochs=50,  # Number of training epochs
        batch=32,
        plots=True,
    )

    print("Training results:")
    print(train_results)

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    print("Evaluation results:")
    print(metrics)

if __name__ == "__main__":
    train_yolo()