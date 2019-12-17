# -*- coding:utf-8 -*-
from model.backbone.light_xception import LightXception
from model.backbone.separable_xception import Xception
from model.backbone.native_xception import NativeXception


def build_encoder(output_stride, BatchNorm, backbone):
    print('Using ' + backbone)
    if backbone == 'light_xception':
        return LightXception(output_stride, BatchNorm)
    elif backbone == 'separable_xception':
        return Xception(output_stride, BatchNorm)
    elif backbone == 'native_xception':
        return NativeXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError
