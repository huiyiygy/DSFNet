# -*- coding:utf-8 -*-
"""
@function: 统计模型计算量和参数量
@author:HuiYi or 会意
@file:flops_counter.py
@time:2019/11/15 12:56
"""
import torch
from ptflops import get_model_complexity_info


def get_flops_and_params(model_def, input_shape=(3, 1025, 513)):
    with torch.cuda.device(0):
        net = model_def()
        flops, params = get_model_complexity_info(net, input_shape, as_strings=True, print_per_layer_stat=True)
        print('Flops:  ' + flops)
        print('Params: ' + params)


if __name__ == "__main__":
    from torchvision import models
    get_flops_and_params(models.shufflenet_v2_x1_0)
