# -*- coding:utf-8 -*-
"""
@function: 批量推理图片
@author:HuiYi or 会意
@file:demo.py
@time:2020/05/15 17:20
"""
import os
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from torch.utils import data
from torchvision import transforms

from dataloader import custom_transforms as tr
from dataloader.utils import DataloaderX as DataLoader
from dataloader.utils import decode_seg_map_sequence
from model.dsfnet import DSFNet


def save_predicted_image(pred, img_path, save_folder):
    rgb = decode_seg_map_sequence(pred)
    num = rgb.shape[0]
    for i in range(num):
        img = Image.fromarray(rgb[i], mode='RGB')
        path = os.path.join(save_folder, os.path.basename(img_path[i]))
        img.save(path)


class ImagesDataset(data.Dataset):
    def __init__(self, root_dir, suffix=".jpg"):
        self.mean = (0.485, 0.456, 0.406)  # for Cityscapes Dataset
        self.std = (0.229, 0.224, 0.225)

        self.root_dir = root_dir
        self.files = sorted(self.recursive_glob(rootdir=self.root_dir, suffix=suffix))

        print("Found %d JPG images" % (len(self.files)))

    @staticmethod
    def recursive_glob(rootdir='.', suffix=''):
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index].rstrip()
        _img = Image.open(img_path).convert('RGB')

        # Normalize
        img = np.array(_img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        # ToTensor
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        sample = {'image': img, 'img_path': img_path}
        return sample


def vis(args):
    if not os.path.exists(args.images_dir):
        raise RuntimeError("=> no images folder found at '{}'".format(args.images_dir))
    images_set = ImagesDataset(args.images_dir, suffix=".jpg")
    images_loader = DataLoader(images_set, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)

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

    if not os.path.exists(args.checkpoint_dir):
        raise RuntimeError("=> no checkpoint folder found at '{}'".format(args.checkpoint_dir))
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint at '{}'".format(args.checkpoint_dir))

    model.eval()

    save_folder = None
    if args.output_dir is None:
        save_folder = args.images_dir + '_predict'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    tbar = tqdm(images_loader, desc='\r')
    for i, sample in enumerate(tbar):
        image, img_path = sample['image'], sample['img_path']
        if args.cuda:
            image = image.cuda()
        with torch.no_grad():
            output = model(image)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # save predicted result
        save_predicted_image(pred, img_path, save_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--backbone', type=str, default='light_xception',
                        choices=['separable_xception', 'light_xception', 'native_xception'], help='backbone name (default: light_xception)')
    parser.add_argument('--out-stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--use-attention', action='store_true', default=True)
    parser.add_argument('--use-channel-shuffle', action='store_true', default=False,
                        help='Only for light_xception, whether to use channel shuffle in DSFNet (default: False)')
    parser.add_argument('--checkpoint-dir', type=str, default='D:/49_light_xception',
                        help='put the path to checkpoint folder')
    parser.add_argument('--images-dir', type=str, default='D:/VID_20200702_162903',
                        help='put the path to images folder')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='images predict results folder. if None, the directory will be --images-dir+_predict')

    vis(parser.parse_args())
