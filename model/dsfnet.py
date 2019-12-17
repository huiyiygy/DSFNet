# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:dsfnet.py
@time:2019/11/17 15:43
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import build_encoder
from model.decoder import build_decoder
from model.sync_batchnorm import SynchronizedBatchNorm2d


class DSFNet(nn.Module):
    def __init__(self, output_stride=8, num_classes=19, sync_bn=False, use_attention=False, backbone='light_xception'):
        """
        Inputs:
        -------
        - output_stride: 8 or 16
        - num_classes: 19
        - sync_bn: whether use SyncBatchNorm
        """
        super(DSFNet, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d  # 用于多卡训练的BN
        else:
            BatchNorm = nn.BatchNorm2d

        self.encoder = build_encoder(output_stride, BatchNorm, backbone)
        self.decoder = build_decoder(num_classes, BatchNorm, use_attention, backbone)

    def forward(self, inputs):
        x, low_level_feat = self.encoder(inputs)
        x = self.decoder(x, low_level_feat)
        # 双线性插值恢复输入图像尺寸
        x = F.interpolate(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)
        return x


if __name__ == "__main__":
    model = DSFNet(output_stride=16, use_attention=True)
    model.eval()
    inp = torch.rand(1, 3, 512, 512)
    output = model(inp)
    print(output.size())

    # (3, 512, 512)
    # LightXception os=8, attention=False FLOPs: 2.62 GMac Params: 503.17 k
    # LightXception os=8, attention=True  FLOPs: 2.74 GMac Params: 794.1 k
    # from utils.flops_counter import get_flops_and_params
    # get_flops_and_params(DSFNet)
    # SeparableXception os=8, attention=False FLOPs: 5.47 GMac Params: 925.1 k
    # SeparableXception os=8, attention=True  FLOPs: 7.07 GMac Params: 4.54 M
    # NativeXception os=8, attention=False FLOPs: 13.1 GMac Params: 7.02 M
    # NativeXception os=8, attention=True  FLOPs: 14.71 GMac Params: 10.64 M
