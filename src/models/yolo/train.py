from ultralytics import YOLO  #type: ignore
from pathlib import Path
import tensorboard

def load_model():
    model_path = "./models/yolo11s.pt"
    #model_path = Path("./models/yolo_experiments/train/weights/last.pt")

    if Path(model_path).exists():
        print("Loading existing model")
        model = YOLO(model_path)
    else:
        print("Model does not exist, will load new model")
        model = YOLO("yolo11s.pt")

    return model

def train_yolo(model):
    train_results = model.train(
        data="./data/zod_yolo/dataset.yaml",  # Path to dataset configuration file
        project="./models/yolo/",
        name="train",
        amp=False,
        save=True,
        save_period=5,
        #resume=True,   # Resuming training
        imgsz=512,
        pretrained=True,
        seed=43,
        val=False,
        fraction=0.8,
        device=0,
        patience=10,
        epochs=5,  # Number of training epochs
        batch=64,
        plots=True,
        exist_ok=True,
        lr0=0.001,
    )

    print("Training results:")
    print(train_results)
    return train_results

def eval_yolo(model):
    # Evaluate the model's performance on the validation set
    metrics = model.val()

    print("Evaluation results:")
    print(metrics)
    return metrics

if __name__ == "__main__":
    model = load_model()
    train_yolo(model)
    #eval_yolo(model)