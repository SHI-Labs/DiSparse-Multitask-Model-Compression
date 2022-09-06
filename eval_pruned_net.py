################################################################################################
# Evaluate Pruned Net Performance
################################################################################################

import os
import random
import argparse
import warnings
import copy

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from prune_utils import *
from dataloaders import *
from scene_net import *
from evaluation import *
################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='dataset: choose between nyuv2, cityscapes, taskonomy', default="nyuv2")
    parser.add_argument('--method', type=str,
                        help='method name', default="disparse_static")
    parser.add_argument('--ratio',type=int,
                        help='percentage of sparsity level', default=90)
    parser.add_argument('--model-path', type=str,
                        help='path to the saved model folder', default="/data/alexsun/save_model/to_release/")
    args = parser.parse_args()

    ################################################################################################
    ratio = args.ratio
    method = args.method
    dataset = args.dataset
    model_folder_path = args.model_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network_name = f"best_{dataset}_{method}_{ratio}"
    save_path = f"{args.model_path}/{network_name}.pth"
    log_file = open(f"logs/{network_name}.txt", "w")

    if dataset == "nyuv2":
        from config_nyuv2 import *
        train_dataset = NYU_v2(DATA_ROOT, 'train', crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = 8, shuffle=True, pin_memory=True)
        test_dataset = NYU_v2(DATA_ROOT, 'test')
        test_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 8, shuffle=True, pin_memory=True)
    elif dataset == "cityscapes":
        from config_cityscapes import *
        train_dataset = CityScapes(DATA_ROOT, 'train', crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = 8, shuffle=True, pin_memory=True)
        test_dataset = CityScapes(DATA_ROOT, 'test')
        test_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 8, shuffle=True, pin_memory=True)
    elif dataset == "taskonomy":
        from config_taskonomy import *
        train_dataset = Taskonomy(DATA_ROOT, 'train', crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = 8, shuffle=True, pin_memory=True)
        test_dataset = Taskonomy(DATA_ROOT, 'test')
        test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, num_workers = 8, shuffle=False, pin_memory=True)
    else:
        print("Unrecognized Dataset Name.")
        exit()

    print("TrainDataset:", len(train_dataset))
    print("TestDataset:", len(test_dataset))
    ################################################################################################
    net = SceneNet(TASKS_NUM_CLASS).to(device)
    # Load Pruned Model       
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module = prune.identity(module, 'weight')
    if dataset == "taskonomy":
        net = nn.DataParallel(net, device_ids=[0, 1])
        net.load_state_dict(torch.load(save_path))
    else:
        net.load_state_dict(torch.load(save_path))
    # Add the mask
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module.weight = module.weight_orig * module.weight_mask
    print_sparsity(net)
    net.eval()

    ######################################################################################################
    warnings.filterwarnings('ignore')
    evaluator = SceneNetEval(device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
    res = evaluator.get_final_metrics(net, test_loader)
    log_file.write(str(res))
    print(res)
    log_file.close()


