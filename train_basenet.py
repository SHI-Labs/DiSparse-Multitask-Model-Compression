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
from train import train
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--network_name', default='nyu_seg_sn', type=str, help='Name of the Network.')
parser.add_argument('--dataset', type=str, help='dataset: choose between nyuv2, cityscapes, taskonomy', default="nyuv2")
parser.add_argument('--dest', default='/data/alexsun/save_model/release_test/', type=str, help='Destination Save Folder.')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network_name = args.network_name
dest = args.dest
dataset = args.dataset

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
if dataset == "taskonomy":
    net = nn.DataParallel(net, device_ids=[0, 1])

criterion = SceneNetLoss(dataset, TASKS, TASKS_NUM_CLASS, LAMBDAS, device, DATA_ROOT)
optimizer = torch.optim.Adam(net.parameters(), lr = INIT_LR, weight_decay = WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_LR_FREQ, gamma=DECAY_LR_RATE)

batch_update = 16
net = train(net, dataset, criterion, optimizer, scheduler, train_loader, test_loader, network_name, batch_update, max_iters=MAX_ITERS, log_file=log_file, save_model=True, method="baseline", dest=dest)
torch.save(net.state_dict(), f"{dest}/final_{network_name}.pth")

######################################################################################################
from evaluation import SceneNetEval
import warnings
import copy
warnings.filterwarnings('ignore')
evaluator = SceneNetEval(device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
net.load_state_dict(torch.load(f"{dest}/best_{network_name}.pth"))
net.eval()
res = evaluator.get_final_metrics(net, test_loader)
log_file.write(str(res))
print(res)
log_file.close()




