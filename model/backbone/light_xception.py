# -*- coding:utf-8 -*-
"""
@function: 基于Depthwise Separable Factorization Convolution的Xception
@author:HuiYi or 会意
@file:light_xception.py
@time:2019/11/15 13:28
"""
import math
import torch
import torch.nn as nn

from model.sync_batchnorm import SynchronizedBatchNorm2d


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    # reshape
    x = x.view(batchsize, groups, num_channels // groups, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class DSFConv(nn.Module):
    """
    DSFConv: Depthwise Separable Factorization Conv
    """
    def __init__(self, inplanes, outplanes, stride=1, dilation=1, BatchNorm=None):
        super(DSFConv, self).__init__()
        # depth-wise
        # Factorization Conv 1*3 -> 3*1
        self.conv1x3 = nn.Conv2d(inplanes, inplanes, (1, 3), stride=(1, stride), padding=(0, dilation), dilation=(1, dilation), groups=inplanes, bias=True)
        self.conv3x1 = nn.Conv2d(inplanes, inplanes, (3, 1), stride=(stride, 1), padding=(dilation, 0), dilation=(dilation, 1), groups=inplanes, bias=True)
        self.bn = BatchNorm(inplanes)
        # point-wise
        self.pointwise = nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv1x3(x)
        x = self.conv3x1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class DSFBlock(nn.Module):
    """
    DSFBlock: Depthwise Separable Factorization Block
    """
    def __init__(self, inplanes, outplanes, stride=1, dilation=1, BatchNorm=None):
        super(DSFBlock, self).__init__()

        self.branch1 = DSFConv(inplanes // 2, outplanes // 2, stride, dilation, BatchNorm)
        self.branch2 = DSFConv(inplanes // 2, outplanes // 2, stride, dilation, BatchNorm)

    def forward(self, x):
        x1, x2 = x.chunk(chunks=2, dim=1)  # 分块拆分

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        x = torch.cat((x1, x2), 1)  # 分块合并
        x = channel_shuffle(x, 2)  # 分块Shuffle

        return x


class XceptionBlock(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None, start_with_relu=True, grow_first=True, is_last=False):
        super(XceptionBlock, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                BatchNorm(planes))
        else:
            self.skip = None

        self.relu = nn.ReLU(True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(DSFBlock(inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(DSFBlock(filters, filters, dilation=dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(DSFBlock(inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(DSFBlock(planes, planes, stride=stride, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(DSFBlock(planes, planes, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp
        x = x + skip
        return x


class LightXception(nn.Module):
    """
    LightXception
    """
    def __init__(self, output_stride=8, BatchNorm=nn.BatchNorm2d):
        """
        Inputs:
        -------
        - output_stride: 8 or 16
        - BatchNorm: SyncBatchNorm or nn.BatchNorm2d
        """
        super(LightXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = [1, 1, 1]
            exit_block_dilations = [1, 5, 7, 9]
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = [5, 7, 9]
            exit_block_dilations = [2, 5, 7, 9]
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = DSFConv(3, 32, stride=2, BatchNorm=BatchNorm)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = DSFBlock(32, 32, stride=1, BatchNorm=BatchNorm)
        self.bn2 = BatchNorm(32)

        self.block1 = XceptionBlock(32, 64, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = XceptionBlock(64, 64, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block3 = XceptionBlock(64, 128, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm, is_last=True)

        # Middle flow
        self.block4 = XceptionBlock(128, 128, reps=3, stride=1, dilation=middle_block_dilation[0], BatchNorm=BatchNorm)
        self.block5 = XceptionBlock(128, 128, reps=3, stride=1, dilation=middle_block_dilation[1], BatchNorm=BatchNorm)
        self.block6 = XceptionBlock(128, 128, reps=3, stride=1, dilation=middle_block_dilation[2], BatchNorm=BatchNorm)
        self.block7 = XceptionBlock(128, 128, reps=3, stride=1, dilation=middle_block_dilation[0], BatchNorm=BatchNorm)
        self.block8 = XceptionBlock(128, 128, reps=3, stride=1, dilation=middle_block_dilation[1], BatchNorm=BatchNorm)
        self.block9 = XceptionBlock(128, 128, reps=3, stride=1, dilation=middle_block_dilation[2], BatchNorm=BatchNorm)
        self.block10 = XceptionBlock(128, 128, reps=3, stride=1, dilation=middle_block_dilation[0], BatchNorm=BatchNorm)
        self.block11 = XceptionBlock(128, 128, reps=3, stride=1, dilation=middle_block_dilation[1], BatchNorm=BatchNorm)

        # Exit flow
        self.block12 = XceptionBlock(128, 256, reps=2, stride=1, dilation=exit_block_dilations[0], BatchNorm=BatchNorm, grow_first=False, is_last=True)

        self.conv3 = DSFBlock(256, 256, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(256)

        self.conv4 = DSFBlock(256, 256, dilation=exit_block_dilations[2], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(256)

        self.conv5 = DSFBlock(256, 256, dilation=exit_block_dilations[3], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(256)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feature = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        # Exit flow
        x = self.block12(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feature

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    model = LightXception(output_stride=8, BatchNorm=nn.BatchNorm2d)
    inputs = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(inputs)
    print(output.size())  # [1, 256, 64, 64]
    print(low_level_feat.size())  # [1, 64, 128, 128]

    # visualize the architecture of LightXception
    # from torch.utils.tensorboard import SummaryWriter
    # with SummaryWriter('../../checkpoint', comment='Xception') as w:
    #     w.add_graph(model, inputs)

    # (3, 512, 512)
    # output_stride=8 Flops:  2.53 GMac Params: 493.95 k
    # from utils.flops_counter import get_flops_and_params
    # get_flops_and_params(LightXception)
