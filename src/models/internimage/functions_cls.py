import argparse
import os
import torch 
from torchvision import transforms
import cv2
import numpy as np
import mmcv
import sys

from config_me import get_config
from InternImage.classification.models import build_model
import torch.nn.functional as F

def args_clsModel(config, pesos):
    parser = argparse.ArgumentParser('Get Intermediate Layer Output')
    parser.add_argument('--cfg', type=str, default=config, help='Path to config file')
    parser.add_argument('--resume', default=pesos, help='resume from checkpoint')
    
    args, unparsed = parser.parse_known_args()
    
    config = get_config(args)
    return args, config

def initialize_model(config, pesos):
    
    _, config_f = args_clsModel(config, pesos)

    model = build_model(config_f)
    checkpoint = torch.load(config_f.MODEL.RESUME, map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=False)
    
    model.cuda()
    model.eval()
    
    return model

def classification_oneDet(model, img):
    
    convert_tensor = transforms.ToTensor()
    tensor_image  = convert_tensor(img).unsqueeze(0).cuda(non_blocking=True)    
    output = model(tensor_image)

    topk=(1, 5)
    maxk = min(max(topk), output.size()[1])
    scores, pred = output.topk(maxk, 1, True, True)
    
    score_list = F.softmax(scores, dim=1).tolist()
    score_h = score_list[0]
    pred = pred.t()
    
    # import pdb; pdb.set_trace()
    # Convert model output to class
    predicted_classes = torch.argmax(output, dim=1)
    # Optionally, if you need the predicted class labels as a list
    predicted_classes_list = predicted_classes.tolist()

    return predicted_classes_list[0], score_h[0]

