# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class TemporalShift(nn.Module):
    """Temporal shift module.
    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_
    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """
 
    def __init__(self, net, num_segments=8, shift_div=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div
 
    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.shift(x, self.num_segments, shift_div=self.shift_div)
        return self.net(x)
 
    @staticmethod
    def shift(x, num_segments, shift_div=8):
        """Perform temporal shift operation on the feature.
        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.
        Returns:
            torch.Tensor: The shifted feature.
        """
        # 假设当前feature map的通道是256，shift_div=3,
        # 那么就有256/3的特征进行shift left,256/3的特征进行shift right，其他一部分特征不动
        # num_segments每个视频采样的帧数
        # 每帧有c个通道，
        # [
        # [0_1,0_2,0_3,1_1,1_2,3_5,3_6,3_7]  第一帧,8个通道，但是shift_div表示这个通道维度被切分成3个等分
        # []  第二帧
        # []  第三帧
        # ]
        # [N, C, H, W]
        n, c, h, w = x.size()
        
        num_segments = n
 
        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, h * w)
 
        # get shift fold
        fold = c // shift_div
 
        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]
 
        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing
 
        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)
 
        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)
 
        # right_split: no shift
 
        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)
 
        # [N, C, H, W]
        # restore the original dimension
        return out.view(n, c, h, w)



class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        
        self.conv1 = TemporalShift(self.conv1, num_segments=8, shift_div=8)
        
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(
        self, input_dim=3, output_dim=128, stride=8, norm_fn="batch", dropout=0.0
    ):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = 64

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.shallow = False
        if self.shallow:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)

            self.conv2 = nn.Conv2d(
                128 + 128 + 96 + 64,
                output_dim * 2,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
            )
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c], dim=1))
        else:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            d = F.interpolate(
                d,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c, d], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x


