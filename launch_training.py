################################################################################################
# Training the Baseline(Unpruned/Unsparsified) Network
################################################################################################

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from dataloaders import *
import matplotlib.pyplot as plt
from scene_net import *
from loss import SceneNetLoss, DiSparse_SceneNetLoss
from train import train, disparse_dynamic_train
import argparse

from evaluation import SceneNetEval
import warnings
import copy

################################################################################################
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--method', type=str, help='method name: baseline, disparse_static, disparse_pt, disparse_dynamic', default="disparse_static")
    parser.add_argument('--ratio',type=int, help='percentage of sparsity level', default=90)
    parser.add_argument('--dataset', type=str, help='dataset: choose between nyuv2, cityscapes, taskonomy', default="nyuv2")
    parser.add_argument('--dest', default='/data/alexsun/save_model/release_test/', type=str, help='Destination Save Folder.')
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dest = args.dest
    dataset = args.dataset
    method = args.method
    ratio = args.ratio
    pruned = method in ["disparse_static", "disparse_pt"]
    pretrained = method in ["disparse_pt"]

    if method == "baseline":
        network_name = f"{dataset}_{method}"
    else:
        network_name = f"{dataset}_{method}_{ratio}"

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

    net = SceneNet(TASKS_NUM_CLASS).to(device)

    # Initialize and Load Pruned Network
    if pruned:
        save_path = f"{dest}/{network_name}.pth"
        import torch.nn.utils.prune as prune
        import torch.nn.functional as F
        from prune_utils import *
        for module in net.modules():
            # Check if it's basic block
            if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
                module = prune.identity(module, 'weight')
        net.load_state_dict(torch.load(save_path))
        for module in net.modules():
            # Check if it's basic block
            if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
                module.weight = module.weight_orig * module.weight_mask
        print_sparsity(net)

    if dataset == "taskonomy":
        net = nn.DataParallel(net, device_ids=[0, 1])

    criterion = SceneNetLoss(dataset, TASKS, TASKS_NUM_CLASS, LAMBDAS, device, DATA_ROOT)
    optimizer = torch.optim.Adam(net.parameters(), lr = INIT_LR, weight_decay = WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_LR_FREQ, gamma=DECAY_LR_RATE)

    batch_update = 16
    if method == "disparse_dynamic":
        D = (100 - ratio) / 100
        # prune_rate, end, interval, init_lr, weight_decay, tasks_num_class, tasks
        config_dict = {"prune_rate":PRUNE_RATE, "end":END, "interval":INT, "init_lr":INIT_LR, "weight_decay":WEIGHT_DECAY, "tasks_num_class":TASKS_NUM_CLASS, "tasks":TASKS, "decay_freq": DECAY_LR_FREQ, "decay_rate":DECAY_LR_RATE}
        amp_criterion = DiSparse_SceneNetLoss(dataset, TASKS, TASKS_NUM_CLASS, LAMBDAS, device, DATA_ROOT)
        net = disparse_dynamic_train(net, dataset, criterion, amp_criterion, optimizer, scheduler, train_loader, test_loader, network_name, batch_update, D, config_dict, max_iters=MAX_ITERS, save_model=True, log_file=log_file, method=method)
    elif not pretrained:
        net = train(net, dataset, criterion, optimizer, scheduler, train_loader, test_loader, network_name, batch_update, max_iters=MAX_ITERS, log_file=log_file, save_model=True, method="baseline", dest=dest)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr = RETRAIN_LR, weight_decay = WEIGHT_DECAY)
        net = train(net, dataset, criterion, optimizer, scheduler, train_loader, test_loader, network_name, batch_update, max_iters = RETRAIN_EPOCH, save_model=True, log_file=log_file, method=method, dest=dest)

    torch.save(net.state_dict(), f"{dest}/final_{network_name}.pth")

    ######################################################################################################
    evaluator = SceneNetEval(device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
    net.load_state_dict(torch.load(f"{dest}/best_{network_name}.pth"))
    net.eval()
    res = evaluator.get_final_metrics(net, test_loader)
    log_file.write(str(res))
    print(res)
    log_file.close()




