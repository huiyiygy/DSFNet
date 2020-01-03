# -*- coding:utf-8 -*-
"""
@function: ImageNet预训练模型
@author:HuiYi or 会意
@file:light_xception_imagenet_pretrain.py
@time:2019/11/30 16:32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.light_xception import LightXception


class XceptionClassifier(nn.Module):
    def __init__(self, num_classes=1000, output_stride=8, use_channel_shuffle=False):
        super(XceptionClassifier, self).__init__()

        self.backbone = LightXception(output_stride=output_stride, use_channel_shuffle=use_channel_shuffle)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x, _ = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = XceptionClassifier()
    model.eval()
    inp = torch.rand(1, 3, 512, 512)
    output = model(inp)
    print(output.size())  # [1, 1000]
