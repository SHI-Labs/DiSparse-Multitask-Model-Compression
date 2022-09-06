################################################################################################
# Loss Function used for NYUvs, CityScapes, and Taskonomy
# Contiain both Regular Multitask Loss and Loss used for DiSparse Gradient Calculation
################################################################################################

import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

################################################################################################
# Regular Multitask Training Loss
class SceneNetLoss(nn.Module):
    def __init__(self, dataset, tasks, tasks_num_class, lambdas, device, data_root=None):
        super().__init__()
        self.device = device
        self.tasks = tasks
        self.tasks_num_class = tasks_num_class
        self.lambdas = lambdas
        self.data_root = data_root
        self.dataset = dataset
        if dataset == "taskonomy":
            assert data_root is not None
            weight = torch.from_numpy(np.load(os.path.join(data_root, 'semseg_prior_factor.npy'))).to(self.device).float()
            self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        elif dataset == "nyuv2":
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        elif dataset == "cityscapes":
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            print("Unrecocgnized Dataset.")
            exit()
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
    ################################################################################################
    def forward(self, preds, targets):
        total_loss = 0
        # Set pred and gt targers
        if 'seg' in self.tasks:
            self.seg_pred = preds[self.tasks.index('seg')]
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            self.seg = targets['seg']
            mult = self.lambdas[self.tasks.index('seg')]
            seg_loss = self.get_seg_loss(seg_num_class)
            total_loss += seg_loss*mult
        if 'sn' in self.tasks:
            self.sn_pred = preds[self.tasks.index('sn')]
            self.normal = targets['normal']
            mult = self.lambdas[self.tasks.index('sn')]
            sn_loss = self.get_sn_loss()
            total_loss += sn_loss*mult
        if 'depth' in self.tasks:
            self.depth_pred = preds[self.tasks.index('depth')]
            self.depth = targets['depth']
            if self.dataset == "taskonomy":
                self.depth_mask = targets['depth_mask']
            mult = self.lambdas[self.tasks.index('depth')]
            depth_loss = self.get_depth_loss()
            total_loss += depth_loss*mult
        if 'keypoint' in self.tasks:
            self.keypoint_pred = preds[self.tasks.index('keypoint')]
            self.keypoint = targets['keypoint']
            mult = self.lambdas[self.tasks.index('keypoint')]
            keypoint_loss = self.get_keypoint_loss()
            total_loss += keypoint_loss*mult
        if 'edge' in self.tasks:
            self.edge_pred = preds[self.tasks.index('edge')]
            self.edge = targets['edge']
            mult = self.lambdas[self.tasks.index('edge')]
            edge_loss = self.get_edge_loss()
            total_loss += edge_loss*mult
        
        return total_loss
    ###########################################################################################
    def get_seg_loss(self, seg_num_class):
        prediction = self.seg_pred.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
        batch_size = self.seg_pred.shape[0]
        new_shape = self.seg_pred.shape[-2:]
        seg_resize = F.interpolate(self.seg.float(), size=new_shape)
        gt = seg_resize.permute(0, 2, 3, 1).contiguous().view(-1)
        loss = self.cross_entropy(prediction, gt.long())
        return loss
    ###########################################################################################
    def get_sn_loss(self):
        prediction = self.sn_pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        new_shape = self.sn_pred.shape[-2:]
        sn_resize = F.interpolate(self.normal.float(), size=new_shape)
        gt = sn_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        labels = (gt.max(dim=1)[0] < 255)
        if hasattr(self, 'normal_mask'):
            normal_mask_resize = F.interpolate(self.normal_mask.float(), size=new_shape)
            gt_mask = normal_mask_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = labels and gt_mask.int() == 1

        prediction = prediction[labels]
        gt = gt[labels]

        prediction = F.normalize(prediction)
        gt = F.normalize(gt)

        loss = 1 - self.cosine_similiarity(prediction, gt).mean()
        return loss
    ###########################################################################################
    def get_depth_loss(self):
        new_shape = self.depth_pred.shape[-2:]
        depth_resize = F.interpolate(self.depth.float(), size=new_shape)
        if self.dataset == "cityscapes":
            if hasattr(self, 'depth_mask'):
                depth_mask_resize = F.interpolate(self.depth_mask.float(), size=new_shape)
            binary_mask = (torch.sum(depth_resize, dim=1) > 3 * 1e-5).unsqueeze(1).to(self.device)
            depth_output = self.depth_pred.masked_select(binary_mask)
            depth_gt = depth_resize.masked_select(binary_mask)
            loss = self.l1_loss(depth_output, depth_gt)
            return loss
        elif self.dataset == "taskonomy":
            if hasattr(self, 'depth_mask'):
                depth_mask_resize = F.interpolate(self.depth_mask.float(), size=new_shape)
                binary_mask = (depth_resize != 255) * (depth_mask_resize.int() == 1).to(self.device)
            else:
                raise ValueError('Dataset %s is invalid' % self.dataset)
            depth_output = self.depth_pred.masked_select(binary_mask)
            depth_gt = depth_resize.masked_select(binary_mask)
            loss = self.l1_loss(depth_output, depth_gt)
            return loss
        else:
            return None
    ###########################################################################################
    def get_keypoint_loss(self):
        new_shape = self.keypoint_pred.shape[-2:]
        keypoint_resize = F.interpolate(self.keypoint.float(), size=new_shape)
        if self.dataset == 'taskonomy':
            binary_mask = keypoint_resize != 255
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        keypoint_output = self.keypoint_pred.masked_select(binary_mask)
        keypoint_gt = keypoint_resize.masked_select(binary_mask)
        loss = self.l1_loss(keypoint_output, keypoint_gt)
        return loss
        
    ###########################################################################################
    def get_edge_loss(self, instance=False):
        new_shape = self.edge_pred.shape[-2:]
        edge_resize = F.interpolate(self.edge.float(), size=new_shape)
        if self.dataset == 'taskonomy':
            binary_mask = edge_resize != 255
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        edge_output = self.edge_pred.masked_select(binary_mask)
        edge_gt = edge_resize.masked_select(binary_mask)
        # torch.sum(torch.abs(self.depth_pred - depth_resize) * binary_mask) / torch.nonzero(binary_mask).size(0)
        loss = self.l1_loss(edge_output, edge_gt)
        return loss
        
    
###########################################################################################
# Loss used for the gradient calculation in DiSparse Prune and Analysis
# In forward function, we use cur_task to choose for which task to calculate the gradient
class DiSparse_SceneNetLoss(nn.Module):
    def __init__(self, dataset, tasks, tasks_num_class, lambdas, device, data_root):
#         super(self, SceneNetLoss).__init__()
        super().__init__()
        self.device = device
        self.tasks = tasks
        self.tasks_num_class = tasks_num_class
        self.lambdas = lambdas
        self.data_root = data_root
        self.dataset = dataset
        if dataset == "taskonomy":
            assert data_root is not None
            weight = torch.from_numpy(np.load(os.path.join(data_root, 'semseg_prior_factor.npy'))).to(self.device).float()
            self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        elif dataset == "nyuv2":
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        elif dataset == "cityscapes":
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            print("Unrecocgnized Dataset.")
            exit()
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
    ###########################################################################################
    def forward(self, preds, targets, cur_task):
        total_loss = 0
        # Set pred and gt targers
        if 'seg' in self.tasks and cur_task == 'seg':
            self.seg_pred = preds[self.tasks.index('seg')]
            seg_num_class = self.tasks_num_class[self.tasks.index('seg')]
            self.seg = targets['seg']
            mult = self.lambdas[self.tasks.index('seg')]
            seg_loss = self.get_seg_loss(seg_num_class)
            return seg_loss

        if 'sn' in self.tasks and cur_task == 'sn':
            self.sn_pred = preds[self.tasks.index('sn')]
            self.normal = targets['normal']
            mult = self.lambdas[self.tasks.index('sn')]
            sn_loss = self.get_sn_loss()
            return sn_loss

        if 'depth' in self.tasks and cur_task == 'depth':
            self.depth_pred = preds[self.tasks.index('depth')]
            self.depth = targets['depth']
            if self.dataset == "taskonomy":
                self.depth_mask = targets['depth_mask']
            mult = self.lambdas[self.tasks.index('depth')]
            depth_loss = self.get_depth_loss()
            return depth_loss
        
        if 'keypoint' in self.tasks and cur_task == 'keypoint':
            self.keypoint_pred = preds[self.tasks.index('keypoint')]
            self.keypoint = targets['keypoint']
            mult = self.lambdas[self.tasks.index('keypoint')]
            keypoint_loss = self.get_keypoint_loss()
            return keypoint_loss
            
        if 'edge' in self.tasks and cur_task == 'edge':
            self.edge_pred = preds[self.tasks.index('edge')]
            self.edge = targets['edge']
            mult = self.lambdas[self.tasks.index('edge')]
            edge_loss = self.get_edge_loss()
            return edge_loss
            
        return total_loss
    ###########################################################################################
    def get_seg_loss(self, seg_num_class, instance=False):

        prediction = self.seg_pred.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
        batch_size = self.seg_pred.shape[0]
        new_shape = self.seg_pred.shape[-2:]
        seg_resize = F.interpolate(self.seg.float(), size=new_shape)
        gt = seg_resize.permute(0, 2, 3, 1).contiguous().view(-1)
        loss = self.cross_entropy(prediction, gt.long())
        return loss
        
    ###########################################################################################
    def get_sn_loss(self):
        prediction = self.sn_pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        new_shape = self.sn_pred.shape[-2:]
        sn_resize = F.interpolate(self.normal.float(), size=new_shape)
        gt = sn_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        labels = (gt.max(dim=1)[0] < 255)
        if hasattr(self, 'normal_mask'):
            normal_mask_resize = F.interpolate(self.normal_mask.float(), size=new_shape)
            gt_mask = normal_mask_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = labels and gt_mask.int() == 1

        prediction = prediction[labels]
        gt = gt[labels]

        prediction = F.normalize(prediction)
        gt = F.normalize(gt)

        loss = 1 - self.cosine_similiarity(prediction, gt).mean()
        return loss
            
    ###########################################################################################
    def get_depth_loss(self):
        new_shape = self.depth_pred.shape[-2:]
        depth_resize = F.interpolate(self.depth.float(), size=new_shape)
        if self.dataset == "cityscapes":
            if hasattr(self, 'depth_mask'):
                depth_mask_resize = F.interpolate(self.depth_mask.float(), size=new_shape)
            binary_mask = (torch.sum(depth_resize, dim=1) > 3 * 1e-5).unsqueeze(1).to(self.device)
            depth_output = self.depth_pred.masked_select(binary_mask)
            depth_gt = depth_resize.masked_select(binary_mask)
            loss = self.l1_loss(depth_output, depth_gt)
            return loss
        elif self.dataset == "taskonomy":
            if hasattr(self, 'depth_mask'):
                depth_mask_resize = F.interpolate(self.depth_mask.float(), size=new_shape)
                binary_mask = (depth_resize != 255) * (depth_mask_resize.int() == 1).to(self.device)
            else:
                raise ValueError('Dataset %s is invalid' % self.dataset)
            depth_output = self.depth_pred.masked_select(binary_mask)
            depth_gt = depth_resize.masked_select(binary_mask)
            loss = self.l1_loss(depth_output, depth_gt)
            return loss
        else:
            return None
    ###########################################################################################
    def get_keypoint_loss(self):
        new_shape = self.keypoint_pred.shape[-2:]
        keypoint_resize = F.interpolate(self.keypoint.float(), size=new_shape)
        if self.dataset == 'taskonomy':
            binary_mask = keypoint_resize != 255
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        keypoint_output = self.keypoint_pred.masked_select(binary_mask)
        keypoint_gt = keypoint_resize.masked_select(binary_mask)
        loss = self.l1_loss(keypoint_output, keypoint_gt)
        return loss
        
    ###########################################################################################
    def get_edge_loss(self, instance=False):
        new_shape = self.edge_pred.shape[-2:]
        edge_resize = F.interpolate(self.edge.float(), size=new_shape)
        if self.dataset == 'taskonomy':
            binary_mask = edge_resize != 255
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        edge_output = self.edge_pred.masked_select(binary_mask)
        edge_gt = edge_resize.masked_select(binary_mask)
        loss = self.l1_loss(edge_output, edge_gt)
        return loss