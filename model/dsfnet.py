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
    def __init__(self, output_stride=8, num_classes=19, sync_bn=False, use_attention=False, backbone='light_xception',
                 use_channel_shuffle=False, pretrained=False, pretrain_file=None):
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

        self.encoder = build_encoder(output_stride, BatchNorm, backbone, use_channel_shuffle, pretrained, pretrain_file)
        self.decoder = build_decoder(num_classes, BatchNorm, use_attention, backbone, use_channel_shuffle)

    def forward(self, inputs):
        x, low_level_feat = self.encoder(inputs)
        x = self.decoder(x, low_level_feat)
        # 双线性插值恢复输入图像尺寸
        x = F.interpolate(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)
        return x

    def get_1x_lr_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    # model = DSFNet(output_stride=16, use_attention=True)
    # model.eval()
    # inp = torch.rand(1, 3, 512, 512)
    # output = model(inp)
    # print(output.size())

    from utils.flops_counter import get_flops_and_params
    get_flops_and_params(DSFNet)
    # (3, 512, 512)
    # LightXception os=8, attention=False FLOPs: 2.62 GMac Params: 503.17 k
    # LightXception os=8, no channel attention FLOPs: 2.74 GMac Params: 723.13 k
    # LightXception os=8, no spatial attention FLOPs: 2.62 GMac Params: 574.13 k
    # LightXception os=8, attention=True  FLOPs: 2.74 GMac Params: 794.1 k

    # LightXception No channel shuffle os=8, attention=False FLOPs: 4.51 GMac Params: 889.12 k
    # LightXception os=8, No channel shuffle os=8, no channel attention FLOPs: 4.72 GMac Params: 1.31 M
    # LightXception os=8, No channel shuffle os=8, no spatial attention FLOPs: 4.51 GMac Params: 960.09 M
    # LightXception No channel shuffle os=8, attention=True  FLOPs: 4.72 GMac Params: 1.38 M

    # SeparableXception os=8, attention=False FLOPs: 5.1 GMac Params: 902.83 k
    # SeparableXception os=8, no channel attention FLOPs: 5.31 GMac Params: 1.32 M
    # SeparableXception os=8, no spatial attention FLOPs:5.1 GMac Params: 0.97 M
    # SeparableXception os=8, attention=True  FLOPs: 5.31 GMac Params: 1.39 M

    # NativeXception os=8, attention=False FLOPs: 34.57 GMac Params: 7.02 M
    # NativeXception os=8, no channel attention FLOPs: 36.18 GMac Params: 10.57 M
    # NativeXception os=8, no spatial attention FLOPs: 34.57 GMac Params: 7.09 M
    # NativeXception os=8, attention=True  FLOPs: 36.18 GMac Params: 10.64 M
