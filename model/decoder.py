# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:decoder.py
@time:2019/11/17 16:03
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.light_xception import DSFBlock, DSFConv
from model.backbone.separable_xception import SeparableConv2d
from model.sync_batchnorm import SynchronizedBatchNorm2d


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, BatchNorm=None):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            BatchNorm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, BatchNorm=None):
        super(SeparableConvBnRelu, self).__init__()

        self.conv = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, kernel_size, stride, padding, BatchNorm=BatchNorm),
            BatchNorm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DSFConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, BatchNorm=None):
        super(DSFConvBnRelu, self).__init__()

        self.conv = nn.Sequential(
            DSF(in_ch, out_ch, stride, dilation, BatchNorm=BatchNorm),
            BatchNorm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class NativeDecoder(nn.Module):
    """
    原生解码器模块
    编码器最后一层特征图上采样后与编码器中间下采样1/4的特征图融合，然后再次上采样
    """

    def __init__(self, num_classes, BatchNorm):
        super(NativeDecoder, self).__init__()

        self.low_level_feat_branch = nn.Conv2d(64, num_classes, 1)
        self.conv = nn.Conv2d(256, num_classes, 1)

        self.last_conv = nn.Sequential(Block(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                       Block(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                       nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1)
                                       )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_feat_branch(low_level_feat)
        x = self.conv(x)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SpatialAttention(nn.Module):
    """
    Spatial Attention Branch
    Residual Attention Network
    """

    def __init__(self, in_ch, num_classes, BatchNorm=None):
        super(SpatialAttention, self).__init__()

        self.down1 = Block(in_ch, in_ch, stride=2, BatchNorm=BatchNorm)
        self.down2 = Block(in_ch, in_ch, stride=2, BatchNorm=BatchNorm)
        self.down3 = Block(in_ch, in_ch, stride=2, BatchNorm=BatchNorm)

        self.conv1 = Block(in_ch, in_ch, stride=1, BatchNorm=BatchNorm)
        self.conv2 = Block(in_ch, in_ch, stride=1, BatchNorm=BatchNorm)
        self.conv3 = Block(in_ch, in_ch, stride=1, BatchNorm=BatchNorm)

        self.last_conv = nn.Sequential(nn.Conv2d(in_ch, num_classes, 1),
                                       BatchNorm(num_classes),
                                       nn.Sigmoid()
                                       )

    def forward(self, x):
        x1 = self.down1(x)  # (32, 32) or (16, 16)
        x2 = self.down2(x1)  # (16, 16) or (8, 8)
        x3 = self.down3(x2)  # (8, 8) or (4, 4)

        x3 = self.conv3(x3)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)

        x2 = self.conv2(x2)
        x2 = x2 + x3
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x1 = self.conv1(x1)
        x1 = x1 + x2
        x1 = F.interpolate(x1, size=x.size()[2:], mode='bilinear', align_corners=True)

        x = self.last_conv(x1)
        return x


class AttentionDecoder(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super(AttentionDecoder, self).__init__()
        self.low_level_feat_branch = nn.Conv2d(64, num_classes, 1)
        self.conv = nn.Conv2d(256, num_classes, 1)

        self.channel_attention_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                      ConvBnRelu(256, 256, kernel_size=1, padding=0, BatchNorm=BatchNorm),
                                                      nn.Conv2d(256, num_classes, 1),
                                                      BatchNorm(num_classes),
                                                      nn.Sigmoid()
                                                      )

        self.spatial_attention_branch = SpatialAttention(256, num_classes, BatchNorm)

        self.last_conv = nn.Sequential(Block(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                       Block(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                       nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1)
                                       )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_feat_branch(low_level_feat)

        spatial_attention = self.spatial_attention_branch(x)
        channel_attention = self.channel_attention_branch(x)

        x = self.conv(x)

        x1 = torch.mul(x, spatial_attention)
        x2 = torch.mul(x, channel_attention)
        x = x + x1 + x2

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


Block = DSFConvBnRelu
DSF = DSFConv


def build_decoder(num_classes, BatchNorm, use_attention, backbone, use_channel_shuffle):
    global Block, DSF
    if backbone == 'native_xception':
        Block = ConvBnRelu
    elif backbone == 'light_xception' and use_channel_shuffle:
        DSF = DSFBlock
    elif backbone == 'separable_xception':
        Block = SeparableConvBnRelu
    if not use_attention:
        return NativeDecoder(num_classes, BatchNorm)
    else:
        return AttentionDecoder(num_classes, BatchNorm)
