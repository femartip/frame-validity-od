from rfdetr import RFDETRBase, RFDETRSmall
from pathlib import Path
import tensorboard


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
        epanded_scales=False,
        use_ema=False,
        run_test=False,
        resolution=512,
        #resume=True
    )

    print("Training results:")
    print(train_results)
    return train_results

if __name__ == "__main__":
    model = RFDETRSmall()
    train_rfdetr(model)