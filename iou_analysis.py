################################################################################################
# Analyze IoU
################################################################################################

import os
import random

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import argparse
from torch.autograd import Variable
from dataloaders import *
import matplotlib.pyplot as plt
from scene_net import *
from loss import SceneNetLoss, DiSparse_SceneNetLoss
from train import train
from prune_utils import *
################################################################################################
def amp_density_analysis(net, dataset, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    grads_abs = {}
    for task in tasks:
        grads_abs[task] = []
        
    # Register Hook
    for layer in test_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(hook_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(hook_forward_linear, layer)

    train_iter = iter(train_loader)
    for i in range(num_batches):
        gt_batch = None
        preds = None
        loss = None
        torch.cuda.empty_cache()

        gt_batch = next(train_iter)
        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
            if dataset == "taskonomy":
                if 'depth_mask' in gt_batch.keys():
                    gt_batch["depth_mask"] = Variable(gt_batch["depth_mask"]).cuda()
                else:
                    print("No Depth Mask Existing. Please check")
                    gt_batch["depth_mask"] = Variable(torch.ones(gt_batch["depth"].shape)).cuda()
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
        if "keypoint" in gt_batch:
                gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
        if "edge" in gt_batch:
            gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()
        
        for i, task in enumerate(tasks):
            preds = None
            torch.cuda.empty_cache()
            test_net.zero_grad()
            preds = test_net.forward(gt_batch['img'])
            loss = criterion(preds, gt_batch, cur_task=task)
            loss.backward()
            ct = 0
            
            for name, layer in test_net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if 'backbone' in name or f'task{i+1}' in name:
                        if len(grads_abs[task]) > ct:
                            grads_abs[task][ct] += torch.abs(layer.weight_mask.grad.data)
                        else:
                            grads_abs[task].append(torch.abs(layer.weight_mask.grad.data))
                        ct += 1

    preds = None
    loss = None
    keep_masks = {}
    for task in tasks:
        keep_masks[task] = []
        
    for i, task in enumerate(tasks):
        cur_grads_abs = grads_abs[task]
        all_scores = torch.cat([torch.flatten(x) for x in cur_grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for g in cur_grads_abs:
            keep_masks[task].append(((g / norm_factor) >= acceptable_score).int())

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks[task]])))
    
    
    # Use PyTorch Prune to set hooks
    parameters_to_prune = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            parameters_to_prune.append((layer, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )

    
    idxs = [0] * len(tasks)
    ct = 0
    
    density_dict = {}
    total_elem = 0
    total_agr = 0
    # Copy the masks
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Let's get the intersection
            # Only all tasks agree to prune, we prune
            if 'backbone' in name:
                final_mask = None
                for i, task in enumerate(tasks):
                    if final_mask is None:
                        final_mask = keep_masks[task][ct].data
                    else:
                        final_mask = final_mask + keep_masks[task][ct].data
                cur_agr = torch.sum(final_mask == 2)
                cur_total = torch.sum(final_mask == 2) + torch.sum(final_mask == 1)
                density_dict[name] = (cur_agr / cur_total).item()
                total_elem += cur_total
                total_agr += cur_agr
                ct += 1
                idxs = [x+1 for x in idxs]
                
            elif 'task1' in name:
                task_name = tasks[0]
                idx = idxs[0]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[0] += 1
                
            elif 'task2' in name:
                task_name = tasks[1]
                idx = idxs[1]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[1] += 1

            elif 'task3' in name:
                task_name = tasks[2]
                idx = idxs[2]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[2] += 1

            elif 'task4' in name:
                task_name = tasks[3]
                idx = idxs[3]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[3] += 1

            elif 'task5' in name:
                task_name = tasks[4]
                idx = idxs[4]
                layer.weight_mask.data = keep_masks[task_name][idx].data
                ct += 1
                idxs[4] += 1
            else:
                print(f"Unrecognized Name: {name}!")
    keep_masks = None
    parameters_to_prune = None
    all_scores = None
    final_mask = None
    grads_abs = None
    torch.cuda.empty_cache()
    outs = net.forward(gt_batch['img'])
    return density_dict, (total_agr/total_elem).item()
################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IoU Analysis')
    parser.add_argument('--dataset', type=str, help='dataset: choose between nyuv2, cityscapes, taskonomy', default="nyuv2")
    args = parser.parse_args()
    dataset = args.dataset

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
        test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, num_workers = 8, shuffle=False, pin_memory=True)
    else:
        print("Unrecognized Dataset Name.")
        exit()
        
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = DiSparse_SceneNetLoss(dataset, TASKS, TASKS_NUM_CLASS, LAMBDAS, device, DATA_ROOT)
    from prune_utils import *
    net = SceneNet(TASKS_NUM_CLASS).to(device)
    if dataset == "taskonomy":
        num_batches = 200
    else:
        num_batches = 50
    keep_ratio = 0.078
    backbone_density, ratio = amp_density_analysis(net, dataset, criterion, train_loader, num_batches, keep_ratio, device, tasks=TASKS)
    # Remove the Downsampling at the end. Get intersection of mask on convolusion layers only.
    del backbone_density['backbone.ds.1.0']
    del backbone_density['backbone.ds.2.0']
    del backbone_density['backbone.ds.3.0']
    with open (f"{dataset}_iou.txt", 'w') as f:
        f.write(str(backbone_density))
        f.close()