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

from model.backbone.xception import DSFBlock
from model.sync_batchnorm import SynchronizedBatchNorm2d


class DSFConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, BatchNorm=None):
        super(DSFConvBnRelu, self).__init__()

        self.conv = nn.Sequential(
            DSFBlock(in_ch, out_ch, stride, dilation, BatchNorm=BatchNorm),
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

    def __init__(self, num_classes, BatchNorm, use_dropout):
        super(NativeDecoder, self).__init__()

        self.low_level_feat_branch = nn.Conv2d(64, num_classes, 1)
        self.conv = nn.Conv2d(256, num_classes, 1)

        if use_dropout:
            self.last_conv = nn.Sequential(DSFConvBnRelu(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                           DSFConvBnRelu(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                           nn.Dropout(0.5),
                                           nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1)
                                           )
        else:
            self.last_conv = nn.Sequential(DSFConvBnRelu(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                           DSFConvBnRelu(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
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

    def __init__(self, in_ch, num_classes):
        super(SpatialAttention, self).__init__()

        self.max_pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_ch, num_classes, 1)
        self.conv2 = nn.Conv2d(in_ch, num_classes, 1)
        self.conv3 = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, x):
        down1 = self.max_pool(x)  # (32, 32) or (16, 16)
        down2 = self.max_pool(down1)  # (16, 16) or (8, 8)
        down3 = self.max_pool(down2)  # (8, 8) or (4, 4)

        down3 = self.conv3(down3)
        down3 = F.interpolate(down3, size=down2.size()[2:], mode='bilinear', align_corners=True)

        down2 = self.conv2(down2)
        down2 = down2 + down3
        down2 = F.interpolate(down2, size=down1.size()[2:], mode='bilinear', align_corners=True)

        down1 = self.conv1(down1)
        down1 = down1 + down2
        down1 = F.interpolate(down1, size=x.size()[2:], mode='bilinear', align_corners=True)

        down1 = self.sigmoid(down1)
        return down1


class AttentionDecoder(nn.Module):
    def __init__(self, num_classes, BatchNorm, use_dropout):
        super(AttentionDecoder, self).__init__()
        self.low_level_feat_branch = nn.Conv2d(64, num_classes, 1)
        self.conv = nn.Conv2d(256, num_classes, 1)

        self.channel_attention_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                      nn.Conv2d(256, num_classes, 1),
                                                      nn.Sigmoid()
                                                      )

        self.spatial_attention_branch = SpatialAttention(256, num_classes)

        if use_dropout:
            self.last_conv = nn.Sequential(DSFConvBnRelu(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                           DSFConvBnRelu(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                           nn.Dropout(0.5),
                                           nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1)
                                           )
        else:
            self.last_conv = nn.Sequential(DSFConvBnRelu(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                           DSFConvBnRelu(num_classes * 2, num_classes * 2, BatchNorm=BatchNorm),
                                           nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1)
                                           )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_feat_branch(low_level_feat)

        spatial_attention = self.spatial_attention_branch(x)
        channel_attention = self.channel_attention_branch(x)

        x = self.conv(x)

        x1 = torch.mul(x, spatial_attention)
        x = x + x1
        x2 = torch.mul(x, channel_attention)
        x = x + x2

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


def build_decoder(num_classes, BatchNorm, use_attention, use_dropout):
    if not use_attention:
        return NativeDecoder(num_classes, BatchNorm, use_dropout)
    else:
        return AttentionDecoder(num_classes, BatchNorm, use_dropout)
