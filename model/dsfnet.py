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

from model.backbone.xception import AlignedXception
from model.decoder import build_decoder
from model.sync_batchnorm import SynchronizedBatchNorm2d


class DSFNet(nn.Module):
    def __init__(self, output_stride=16, num_classes=19, sync_bn=False, is_native=True):
        """
        Inputs:
        -------
        - output_stride: 8 or 16
        - num_classes: 19
        - sync_bn: whether use SyncBatchNorm
        - is_native:
            为True时，编码模块需返回中间下采样1/4的特征图，解码模块直接上采样。
            为False时，编码模块只返回x, 解码模块添加注意力分支。
        """
        super(DSFNet, self).__init__()

        self.is_native = is_native
        BatchNorm = None
        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d  # 用于多卡训练的BN
        else:
            BatchNorm = nn.BatchNorm2d

        self.encoder = AlignedXception(output_stride, BatchNorm, is_native)
        self.decoder = build_decoder(num_classes, BatchNorm, is_native)

    def forward(self, inputs):
        if self.is_native:
            x, low_level_feat = self.encoder(inputs)
            x = self.decoder(x, low_level_feat)
        else:
            x = self.encoder(inputs)
            x = self.decoder(x)
        # 双线性插值恢复输入图像尺寸
        x = F.interpolate(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)
        return x


if __name__ == "__main__":
    model = DSFNet(output_stride=16, is_native=True)
    model.eval()
    inp = torch.rand(1, 3, 512, 512)
    output = model(inp)
    print(output.size())

    # (3, 512, 512)
    # output_stride=16, is_native=True FLOPs: 2.52 GMac Params: 579.35 k
    # from utils.flops_counter import get_flops_and_params
    # get_flops_and_params(DSFNet)
