# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:native_xception.py
@time:2019/12/17 10:23
"""
import torch
import torch.nn as nn

from model.sync_batchnorm import SynchronizedBatchNorm2d


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                                      BatchNorm(planes))
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(nn.Conv2d(inplanes, planes, 3, 1, padding=dilation, dilation=dilation))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(nn.Conv2d(filters, filters, 3, 1, padding=dilation, dilation=dilation))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(nn.Conv2d(inplanes, planes, 3, 1, padding=dilation, dilation=dilation))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(nn.Conv2d(planes, planes, 3, 2, padding=1))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(nn.Conv2d(planes, planes, 3, 1, padding=1))
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


class NativeXception(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=nn.BatchNorm2d):
        super(NativeXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = [1, 1, 1]
            exit_block_dilations = [1, 2, 5, 1]  # To be Tested
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = [1, 2, 5]
            exit_block_dilations = [1, 2, 5, 1]
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(32)
        # do relu here
        self.block1 = Block(32, 64, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = Block(64, 64, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block3 = Block(64, 128, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm, is_last=True)

        # Middle flow
        self.block4 = Block(128, 128, reps=3, stride=1, dilation=middle_block_dilation[0], BatchNorm=BatchNorm)
        self.block5 = Block(128, 128, reps=3, stride=1, dilation=middle_block_dilation[1], BatchNorm=BatchNorm)
        self.block6 = Block(128, 128, reps=3, stride=1, dilation=middle_block_dilation[2], BatchNorm=BatchNorm)
        self.block7 = Block(128, 128, reps=3, stride=1, dilation=middle_block_dilation[0], BatchNorm=BatchNorm)
        self.block8 = Block(128, 128, reps=3, stride=1, dilation=middle_block_dilation[1], BatchNorm=BatchNorm)
        self.block9 = Block(128, 128, reps=3, stride=1, dilation=middle_block_dilation[2], BatchNorm=BatchNorm)
        self.block10 = Block(128, 128, reps=3, stride=1, dilation=middle_block_dilation[0], BatchNorm=BatchNorm)
        self.block11 = Block(128, 128, reps=3, stride=1, dilation=middle_block_dilation[1], BatchNorm=BatchNorm)

        # Exit flow
        self.block12 = Block(128, 256, reps=2, stride=1, dilation=exit_block_dilations[0], BatchNorm=BatchNorm,
                             grow_first=False, is_last=True)

        self.conv3 = nn.Conv2d(256, 256, 3, stride=1, padding=exit_block_dilations[1], dilation=exit_block_dilations[1])
        self.bn3 = nn.BatchNorm2d(256)

        # do relu here
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=exit_block_dilations[2], dilation=exit_block_dilations[2])
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=exit_block_dilations[3], dilation=exit_block_dilations[3])
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
    model = NativeXception(output_stride=8, BatchNorm=nn.BatchNorm2d)
    inputs = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(inputs)
    print(output.size())  # [1, 256, 64, 64]
    print(low_level_feat.size())  # [1, 64, 128, 128]

    # (3, 512, 512)
    # output_stride=8 Flops:  12.62 GMac Params: 6.99 M
    # from utils.flops_counter import get_flops_and_params
    # get_flops_and_params(NativeXception)
