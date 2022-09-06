################################################################################################
# Model used For Multitask Experiments
# Dense Prediction
################################################################################################

import torch
import torch.nn as nn
import numpy as np
from scene_net_base_blocks import *
import torch.nn.functional as F

################################################################################################
class Deeplab_ResNet_Backbone(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(zip(filt_sizes, layers, strides, dilations)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.layer_config = layers
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    ################################################################################################
    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    ################################################################################################
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine = True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return layers, downsample
    ################################################################################################
    def forward(self, x, policy=None):
        x = self.seed(x)
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                x = F.relu(residual + self.blocks[segment][b](x))
        return x
    
################################################################################################
class SceneNet(nn.Module):
    def __init__(self, num_classes_tasks, skip_layer=0):
        super().__init__()
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.num_tasks = len(num_classes_tasks)
       
        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

        self.layers = layers
        self.skip_layer = skip_layer

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)
    ################################################################################################
    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'logits' in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ('task' in name and 'logits' in name):
                params.append(param)
        return params
    ################################################################################################

    def forward(self, img, num_train_layers=None, hard_sampling=False, mode='train'):
        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        cuda_device = img.get_device()
        # Use a shared trunk structure. Shared extracted features
        outputs = []
        feats = self.backbone(img)
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats) + \
                     getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats) + \
                     getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats) + \
                     getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats)
            outputs.append(output)
        return outputs
################################################################################################
