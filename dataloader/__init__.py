# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:__init__.py.py
@time:2019/11/14 14:05
"""
from dataloader.datasets import cityscapes
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError
