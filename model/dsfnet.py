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

from model.backbone.xception import Xception
from model.decoder import build_decoder
from model.sync_batchnorm import SynchronizedBatchNorm2d


class DSFNet(nn.Module):
    def __init__(self, output_stride=8, num_classes=19, sync_bn=False, use_attention=False):
        """
        Inputs:
        -------
        - output_stride: 8 or 16
        - num_classes: 19
        - sync_bn: whether use SyncBatchNorm
        """
        super(DSFNet, self).__init__()

        BatchNorm = None
        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d  # 用于多卡训练的BN
        else:
            BatchNorm = nn.BatchNorm2d

        self.encoder = Xception(output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm, use_attention)

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
    # output_stride=8, use_attention=False FLOPs: 2.62 GMac Params: 503.17 k
    # output_stride=8, use_attention=True  FLOPs: 2.74 GMac Params: 794.1 k
    # from utils.flops_counter import get_flops_and_params
    # get_flops_and_params(DSFNet)
