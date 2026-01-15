import os
import os.path as osp
from argparse import Namespace

from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator

from zod.anno.object import OBJECT_CLASSES


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, cfg.RESULT_SUBDIR)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def build_config(args: Namespace) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("valid",)
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(OBJECT_CLASSES)
    cfg.OUTPUT_DIR = "./models/faster_rcnn/"
    cfg.SOLVER.IMS_PER_BATCH = 22
    cfg.SOLVER.BASE_LR = 0.025
    cfg.SOLVER.MAX_ITER = 90000
    cfg.SOLVER.STEPS = (60000, 80000)
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 1000
    if False:  # By default loads pre-trained model for fine-tuning, this will train without pre-loading weights
        cfg.MODEL.WEIGHTS = None
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.WARMUP_FACTOR = 0.1
        cfg.SOLVER.WARMUP_METHOD = "constant"
        cfg.SOLVER.STEPS = [step + cfg.SOLVER.WARMUP_ITERS for step in cfg.SOLVER.STEPS]
        cfg.SOLVER.MAX_ITER += cfg.SOLVER.WARMUP_ITERS
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def register_dataset(zod_path, train_json, val_json, image_root):
    register_coco_instances("train", {}, osp.join(zod_path, "train", train_json), image_root)
    register_coco_instances("valid", {}, osp.join(zod_path, "valid", val_json), image_root)


def main(args: Namespace):
    register_dataset(args.zod_path, args.train_json, args.val_json, args.image_root)
    cfg = build_config(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--zod-path", default="data/zod_coco")
    parser.add_argument("--train-json", default="_annotations.coco.json")
    parser.add_argument("--val-json", default="_annotations.coco.json")
    parser.add_argument("--image-root", default=".")
    
    args = parser.parse_args()
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
