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
    def __init__(self, num_classes, BatchNorm):
        super(NativeDecoder, self).__init__()
        # 64 + 256 = 320 low_level_feat: 64  x: 256
        self.last_conv = nn.Sequential(DSFConvBnRelu(320, 256, BatchNorm=BatchNorm),
                                       nn.Dropout(0.5),
                                       DSFConvBnRelu(256, 256, BatchNorm=BatchNorm),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
                                       )
        self._init_weight()

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, BatchNorm, is_native):
    if is_native:
        return NativeDecoder(num_classes,  BatchNorm)
