import argparse
import os
import os.path as osp
import argparse
from argparse import Namespace
import tensorboard
import wandb

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer

from zod.anno.object import OBJECT_CLASSES
from dotenv import load_dotenv
import os
from huggingface_hub import HfApi, create_repo

load_dotenv()

def upload_to_hub(model_dir: str, repo_name: str, token: str) -> None:
    api = HfApi()
    create_repo(repo_name, token=token, repo_type="model", exist_ok=True)
    #api.upload_folder(repo_id=repo_name, folder_path=model_dir, repo_type="model", token=token, path_in_repo=".", commit_message="Upload Detectron2 Faster R-CNN checkpoint + config")
    
    api.upload_file(path_or_fileobj=f"{model_dir}/config.yaml", path_in_repo="config.yaml", repo_id=repo_name, repo_type="model")
    api.upload_file(path_or_fileobj=f"{model_dir}/MODEL_CARD.md", path_in_repo="README.md",      repo_id=repo_name, repo_type="model")
    # Upload all tensorboard event files
    for filename in os.listdir(model_dir):
        if filename.startswith("events.out.tfevents"):
            api.upload_file(path_or_fileobj=os.path.join(model_dir, filename), path_in_repo=filename, repo_id=repo_name, repo_type="model")
        if filename.endswith(".pth"):
            api.upload_file(path_or_fileobj=os.path.join(model_dir, filename), path_in_repo=filename, repo_id=repo_name, repo_type="model")
    



class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        aug = [
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=aug)   #type: ignore
        return build_detection_train_loader(cfg, mapper=mapper)     #type: ignore

def _get_last_checkpoint(output_dir: str) -> str:
    last_ckpt_path = os.path.join(output_dir, "last_checkpoint")
    if not os.path.isfile(last_ckpt_path):
        raise FileNotFoundError(
            f"Could not find last checkpoint at {last_ckpt_path}. "
            "Train first or pass a valid checkpoint."
        )
    with open(last_ckpt_path, "r") as f:
        last_checkpoint = f.read().strip()
    return os.path.join(output_dir, last_checkpoint)


def build_config(args: Namespace) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("valid",)
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(OBJECT_CLASSES)
    cfg.OUTPUT_DIR = "./models/faster-rcnn/"
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 150000
    cfg.SOLVER.STEPS = (80000, 110000)
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 5000
    if False:  # By default loads pre-trained model for fine-tuning, this will train without pre-loading weights
        cfg.MODEL.WEIGHTS = None
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.WARMUP_FACTOR = 0.1
        cfg.SOLVER.WARMUP_METHOD = "constant"
        cfg.SOLVER.STEPS = [step + cfg.SOLVER.WARMUP_ITERS for step in cfg.SOLVER.STEPS]
        cfg.SOLVER.MAX_ITER += cfg.SOLVER.WARMUP_ITERS
    if args.evaluate_only or args.test_only:
        cfg.MODEL.WEIGHTS = _get_last_checkpoint(cfg.OUTPUT_DIR)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def register_dataset(zod_path, train_json, val_json, test_json, image_root):
    register_coco_instances("train", {}, osp.join(zod_path, train_json), image_root)
    register_coco_instances("valid", {}, osp.join(zod_path, val_json), image_root)
    register_coco_instances("test", {}, osp.join(zod_path, test_json), image_root)


def evaluate_model(cfg: CfgNode) -> None:
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
    evaluator = COCOEvaluator("valid", cfg, False, output_dir=cfg.OUTPUT_DIR)
    metrics = Trainer.test(cfg, model, [evaluator])
    print(metrics)

def test_model(cfg: CfgNode) -> None:
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
    evaluator = COCOEvaluator("test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    metrics = Trainer.test(cfg, model, [evaluator])
    print(metrics)

def main(args: Namespace):
    register_dataset(args.zod_path, args.train_json, args.val_json, args.test_json, args.image_root)
    cfg = build_config(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    if args.evaluate_only:
        evaluate_model(cfg)
        return
    
    if args.test_only:
        test_model(cfg)
        return

    if comm.is_main_process():
        wandb.init(project="faster-rcnn-zod", name="faster-rcnn-finetune", dir=cfg.OUTPUT_DIR, config={"cfg_yaml": cfg.dump()}, sync_tensorboard=True)
        os.environ["WANDB_TENSORBOARD_LOGDIR"] = cfg.OUTPUT_DIR

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--zod-path", default="data/zod_coco")
    parser.add_argument("--train-json", default="_annotations.coco.json")
    parser.add_argument("--val-json", default="_annotations.coco.json")
    parser.add_argument("--test-json", default="_annotations.coco.json")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    
    print("Command Line Args:", args)

    if args.push_to_hub:
        upload_to_hub("./models/faster-rcnn/", "femartip/faster-rcnn-zod", os.environ["HF_TOKEN"])
    else:
        launch(main,1,args=(args,),)
