################################################################################################
# Model Building Blocks
################################################################################################

import torch
import torch.nn as nn
import numpy as np

################################################################################################
def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)

################################################################################################
# No projection: identity shortcut
# conv -> bn -> relu -> conv -> bn
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.planes = planes
        # 4 level channel usage: 0 -- 0%; 1 -- 25 %; 2 -- 50 %; 3 -- 100%
        self.keep_channels = (planes * np.cumsum([0, 0.25, 0.25, 0.5])).astype('int')
        self.keep_masks = []
        for kc in self.keep_channels:
            mask = np.zeros([1, planes, 1, 1])
            mask[:, :kc] = 1
            self.keep_masks.append(mask)
        self.keep_masks = torch.from_numpy(np.concatenate(self.keep_masks)).float()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, affine = True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine = True)

    def forward(self, x, keep=None):
        cuda_device = x.get_device()
        out = self.conv1(x)
        out = self.bn1(out)
        # used for deep elastic
        if keep is not None:
            keep = keep.long()
            bs, h, w = out.shape[0], out.shape[2], out.shape[3]
            # mask: [batch_size, c, 1, 1]
            mask = self.keep_masks[keep].to(cuda_device)
            # mask: [batch_size, c, h, w]
            mask = mask.repeat(1, 1, h, w)
            out = out * mask
        out = self.relu(out)
        out = self.conv2(out)
        y = self.bn2(out)
        return y
        
################################################################################################
class Classification_Module(nn.Module):
    def __init__(self, inplanes, num_classes, rate=12):
        super(Classification_Module, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x
    
