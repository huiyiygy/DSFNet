# -*- coding:utf-8 -*-
"""
@function: 测试集可视化与评估
@author:HuiYi or 会意
@file:vis_eval.py
@time:2019/11/14 14:07
"""
import os
import numpy as np
import torch
from tqdm import tqdm

from argparse import ArgumentParser
from PIL import Image
from dataloader.utils import DataloaderX as DataLoader

from dataloader.datasets import cityscapes
from dataloader.utils import decode_seg_map_sequence
from model.dsfnet import DSFNet
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses


def save_predicted_image(pred, img_path, save_folder):
    rgb = decode_seg_map_sequence(pred)
    num = rgb.shape[0]
    for i in range(num):
        img = Image.fromarray(rgb[i], mode='RGB')
        folder = os.path.join(save_folder, img_path[i].split(os.sep)[-2])
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, os.path.basename(img_path[i]))
        img.save(path)


def main(args):
    test_set = cityscapes.CityscapesSegmentation(args, split='val',  mode='test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    nclass = 19
    model = DSFNet(num_classes=nclass,
                   output_stride=args.out_stride,
                   sync_bn=False,
                   use_attention=args.use_attention,
                   backbone=args.backbone,
                   use_channel_shuffle=args.use_channel_shuffle
                   )
    if not args.no_cuda and torch.cuda.is_available():
        args.cuda = True
        model = model.cuda()
        # 提升Pytorch运行效率
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        args.cuda = False

    if not os.path.exists(args.checkpoint_folder):
        raise RuntimeError("=> no checkpoint folder found at '{}'".format(args.checkpoint_folder))
    checkpoint = torch.load(os.path.join(args.checkpoint_folder, 'model_best.pth.tar'))

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint at '{}'".format(args.checkpoint_folder))

    model.eval()

    criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='ce')
    evaluator = Evaluator(nclass)

    save_folder = None
    if args.is_vis:
        save_folder = os.path.join(args.checkpoint_folder, 'vis_color')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    tbar = tqdm(test_loader, desc='\r')
    test_loss = 0.0
    for i, sample in enumerate(tbar):
        image, target, img_path = sample['image'], sample['label'], sample['img_path']
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)
        loss = criterion(output, target)
        test_loss += loss.item()
        tbar.set_description('Test loss: %.6f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # save predicted result
        if args.is_vis:
            save_predicted_image(pred, img_path, save_folder)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

    pixel_accuracy = evaluator.pixel_accuracy()
    mean_class_pixel_accuracy = evaluator.pixel_accuracy_class()
    per_class_iou, mIoU = evaluator.mean_intersection_over_union()
    FWIoU = evaluator.frequency_weighted_intersection_over_union()
    class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',
                   'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                   'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                   'motorcycle', 'bicycle']
    per_class_iou = dict(zip(class_names, per_class_iou))
    print('test Result:')
    print("pixel_acc:{}, mean_class_pixel_acc:{}, mIoU:{}, fwIoU: {}".format(pixel_accuracy, mean_class_pixel_accuracy, mIoU, FWIoU))
    with open(os.path.join(args.checkpoint_folder, 'val_set_mIOU.txt'), 'w') as f:
        f.write("pixel_accuracy:{}\nmean_class_pixel_accuracy:{}\nmIoU:{}\nfwIoU:{}\nper_class_iou:\n{}".format(pixel_accuracy, mean_class_pixel_accuracy, mIoU, FWIoU, per_class_iou))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--backbone', type=str, default='light_xception',
                        choices=['separable_xception', 'light_xception', 'native_xception'], help='backbone name (default: light_xception)')
    parser.add_argument('--crop-size', type=int, default=512)  # short size , long size = crop size * 2
    parser.add_argument('--out-stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--use-attention', action='store_true', default=False)
    parser.add_argument('--use-channel-shuffle', action='store_true', default=False,
                        help='Only for light_xception, whether to use channel shuffle in DSFNet (default: False)')
    parser.add_argument('--is-vis', action='store_true', default=False,
                        help='whether to save predicted result (default: False)')
    parser.add_argument('--checkpoint-folder', type=str, default=None,
                        help='put the path to checkpoint folder')

    main(parser.parse_args())
