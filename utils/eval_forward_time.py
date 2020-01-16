# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:eval_forward_time.py
@time:2019/11/19 15:55
"""
import torch
import time

from argparse import ArgumentParser
from torch.autograd import Variable

from model.dsfnet import DSFNet

# 提升Pytorch运行效率
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    model = DSFNet(num_classes=19,
                   output_stride=args.out_stride,
                   sync_bn=False,
                   use_attention=args.use_attention,
                   backbone=args.backbone,
                   use_channel_shuffle=args.use_channel_shuffle
                   )
    images = torch.randn(args.batch_size, 3, args.height, args.width)
    if not args.no_cuda and torch.cuda.is_available():
        images = images.cuda()
        model = model.cuda()

    model.eval()

    time_train = []
    i = 0

    while True:
        start_time = time.time()
        images = Variable(images)

        with torch.no_grad():
            outputs = model(images)

        if not args.no_cuda:
            torch.cuda.synchronize(device=0)    # wait for cuda to finish (cuda is asynchronous!)

        if i != 0:  # first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            pre_img_time = fwt / args.batch_size
            mean_time = sum(time_train) / len(time_train) / args.batch_size
            pre_FPS = round(1 / pre_img_time)
            mean_FPS = round(1 / mean_time)
            print("Forward time per img (b=%d): %.4f s (Mean: %.4f s, pre_FPS: %d, mean_FPS: %d)" % (args.batch_size, pre_img_time, mean_time,
                                                                                                     pre_FPS, mean_FPS))

        time.sleep(1)  # to avoid overheating the GPU too much
        i += 1


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--out-stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--use-attention', action='store_true', default=True)
    parser.add_argument('--use-channel-shuffle', action='store_true', default=False,
                        help='Only for light_xception, whether to use channel shuffle in DSFNet (default: False)')
    parser.add_argument('--backbone', type=str, default='light_xception',
                        choices=['separable_xception', 'light_xception', 'native_xception'])

    main(parser.parse_args())
    # DSFNet os 8 attention with channel shuffle FPS:45
    # DSFNet os 8 with spatial attention with channel shuffle FPS:45
    # DSFNet os 8 with channel attention with channel shuffle FPS:46
    # DSFNet os 8 without attention with channel shuffle FPS:46
    # DSFNet os 8 attention without channel shuffle FPS:58
    # DSFNet os 8 no channel attention without channel shuffle FPS:58
    # DSFNet os 8 no spatial attention without channel shuffle FPS:
    # DSFNet os 8 without attention without channel shuffle FPS:62
    # SeparableXception os 8 attention FPS:58
    # SeparableXception os 8 no channel attention FPS:58
    # SeparableXception os 8 no spatial attention FPS:61
    # SeparableXception os 8 without attention FPS:59
    # NativeXception os 8 attention FPS:43
    # NativeXception os 8 no channel attention FPS:44
    # NativeXception os 8 no spatial attention FPS:45
    # NativeXception os 8 without attention FPS:45
