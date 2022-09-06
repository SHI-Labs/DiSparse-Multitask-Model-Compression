################################################################################################
# Create pruned/sparse net used for later finetuning/training
################################################################################################


import os
import random
import argparse
import cv2
import numpy as np
import warnings
import copy

import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

from dataloaders import *
from scene_net import *
from prune_utils import *
from loss import SceneNetLoss, DiSparse_SceneNetLoss

import torch.nn.utils.prune as prune


# **************************************************************************************************************** #
def create_disparse_static_nyuv2(net, ratio, criterion, train_loader, num_batches, device, tasks):
    if ratio == 90:
        keep_ratio = 0.08
    elif ratio == 70:
        keep_ratio = 0.257
    elif ratio == 50:
        keep_ratio = 0.46
    elif ratio == 30:
        keep_ratio = 0.675
    else:
        keep_ratio = (100 - ratio) / 100
    net = disparse_prune_static(net, criterion, train_loader, num_batches, keep_ratio, device, tasks)
    return net

def create_disparse_static_cityscapes(net, ratio, criterion, train_loader, num_batches, device, tasks):
    if ratio == 90:
        keep_ratio = 0.095
    elif ratio == 70:
        keep_ratio = 0.3
    elif ratio == 50:
        keep_ratio = 0.51
    elif ratio == 30:
        keep_ratio = 0.71
    else:
        keep_ratio = (100 - ratio) / 100
    net = disparse_prune_static(net, criterion, train_loader, num_batches, keep_ratio, device, tasks)
    return net

def create_disparse_static_taskonomy(net, ratio, criterion, train_loader, num_batches, device, tasks):
    if ratio == 90:
        keep_ratio = 0.097
    elif ratio == 70:
        keep_ratio = 0.257
    elif ratio == 50:
        keep_ratio = 0.46
    elif ratio == 30:
        keep_ratio = 0.675
    else:
        keep_ratio = (100 - ratio) / 100
    net = disparse_prune_static(net, criterion, train_loader, num_batches, keep_ratio, device, tasks)
    return net    

def create_disparse_pt_nyuv2(net, ratio, criterion, train_loader, num_batches, device, tasks, dest="/data"):
    if ratio == 90:
        keep_ratio = 0.1
    elif ratio == 70:
        keep_ratio = 0.3
    elif ratio == 50:
        keep_ratio = 0.5
    elif ratio == 30:
        keep_ratio = 0.7
    else:
        keep_ratio = (100 - ratio) / 100
    net.load_state_dict(torch.load(f"/data/alexsun/save_model/nyu_v2/final_seg_sn.pth"))
    # net.load_state_dict(torch.load(f"{dest}/final_seg_sn.pth"))
    net = disparse_prune_pretrained_l1(net, criterion, train_loader, num_batches, keep_ratio, device, tasks)
    return net

def create_disparse_pt_cityscapes(net, ratio, criterion, train_loader, num_batches, device, tasks, dest="/data"):
    if ratio == 90:
        keep_ratio = 0.13
    elif ratio == 70:
        keep_ratio = 0.37
    elif ratio == 50:
        keep_ratio = 0.6
    elif ratio == 30:
        keep_ratio = 0.644
    else:
        keep_ratio = (100 - ratio) / 100
    net.load_state_dict(torch.load(f"/data/alexsun/save_model/cityscapes/final_seg_sn.pth"))
    # net.load_state_dict(torch.load(f"{dest}/final_seg_depth.pth"))
    net = disparse_prune_pretrained_l1(net, criterion, train_loader, num_batches, keep_ratio, device, tasks)
    return net

def create_disparse_pt_taskonomy(net, ratio, criterion, train_loader, num_batches, device, tasks, dest="/data"):
    if ratio == 90:
        keep_ratio = 0.2
    elif ratio == 70:
        keep_ratio = 0.3
    elif ratio == 50:
        keep_ratio = 0.5
    elif ratio == 30:
        keep_ratio = 0.7
    else:
        keep_ratio = (100 - ratio) / 100
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(f"/data/alexsun/save_model/final_taskonomy_5task.pth"))
    # net.load_state_dict(torch.load(f"{dest}/final_taskonomy_5task.pth"))
    net = net.module
    net = disparse_prune_pretrained(net, criterion, train_loader, num_batches, keep_ratio, device, tasks)
    return net
# **************************************************************************************************************** #

################################################################################################
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset: choose between nyuv2, cityscapes, taskonomy', default="nyuv2")
    parser.add_argument('--num_batches',type=int, help='number of batches to estimate importance', default=50)
    parser.add_argument('--method', type=str, help='method name', default="disparse_static")
    parser.add_argument('--ratio',type=int, help='percentage of sparsity level', default=90)
    parser.add_argument('--dest', default='/data/alexsun/save_model/release_test/', type=str, help='Destination Save Folder.')
    args = parser.parse_args()

    dataset = args.dataset
    ratio = args.ratio
    num_batches = args.num_batches
    method = args.method
    dest = args.dest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ################################################################################################
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
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE//4, num_workers = 8, shuffle=True, pin_memory=True)
        test_dataset = Taskonomy(DATA_ROOT, 'test')
        test_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 8, shuffle=True, pin_memory=True)
    else:
        print("Unrecognized Dataset Name.")
        exit()
    ################################################################################################
    network_name = f"{dataset}_{method}_{ratio}"
    save_path = f"{dest}/{network_name}.pth"

    ################################################################################################
    net = SceneNet(TASKS_NUM_CLASS).to(device)
    if method == "disparse_static":
        criterion = DiSparse_SceneNetLoss(dataset, TASKS, TASKS_NUM_CLASS, LAMBDAS, device, DATA_ROOT)
        if dataset == "nyuv2":
            net = create_disparse_static_nyuv2(net, ratio, criterion, train_loader, num_batches, device, tasks=TASKS)
        elif dataset == "cityscapes":
            net = create_disparse_static_cityscapes(net, ratio, criterion, train_loader, num_batches, device, tasks=TASKS)
        elif dataset == "taskonomy":
            net = create_disparse_static_taskonomy(net, ratio, criterion, train_loader, num_batches, device, tasks=TASKS)
        else:
            print("Unrecognized Dataset Name.")
            exit()
        
    elif method == "disparse_pt":
        criterion = DiSparse_SceneNetLoss(dataset, TASKS, TASKS_NUM_CLASS, LAMBDAS, device, DATA_ROOT)
        if dataset == "nyuv2":
            net = create_disparse_pt_nyuv2(net, ratio, criterion, train_loader, num_batches, device, tasks=TASKS, dest=dest)
        elif dataset == "cityscapes":
            net = create_disparse_pt_cityscapes(net, ratio, criterion, train_loader, num_batches, device, tasks=TASKS, dest=dest)
        elif dataset == "taskonomy":
            net = create_disparse_pt_taskonomy(net, ratio, criterion, train_loader, num_batches, device, tasks=TASKS, dest=dest)
        else:
            print("Unrecognized Dataset Name.")
            exit()

    print_sparsity(net)
    print(f"Saving the pruned model to {save_path}")
    torch.save(net.state_dict(), save_path)
