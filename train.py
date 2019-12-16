# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:train.py
@time:2019/11/14 14:05
"""
import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloader import make_data_loader
from model.dsfnet import DSFNet
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.calculate_weights import calculate_weigths_labels
from utils.loss import SegmentationLosses
from utils.saver import Saver
from utils.metrics import Evaluator


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.writer = SummaryWriter(self.saver.experiment_dir, comment='LightXception')

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        self.model = DSFNet(num_classes=self.nclass,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            use_attention=args.use_attention
                            )

        # Define Optimizer
        if self.args.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=args.momentum,
                                             weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif self.args.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

        # Define lr scheduler
        if self.args.lr_scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,  mode='max', factor=0.3, verbose=True, min_lr=1e-8)
        elif self.args.lr_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        elif self.args.lr_scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=1e-8)
        else:
            raise NotImplementedError

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
            # 提升Pytorch运行效率
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            # 保证模型的可重复性
            torch.backends.cudnn.deterministic = True

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    def training(self, epoch):
        print('[Epoch: %d, previous best = %.6f]' % (epoch+1, self.best_pred))
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            tbar.set_description('Train loss: %.6f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        # 训练时只计算每个epoch最后一次迭代的准确率，因为训练集数据多，如果统计所有就太慢了。
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        self.evaluator.add_batch(target, pred)

        # Fast test during the training
        pixel_accuracy = self.evaluator.pixel_accuracy()
        mean_class_pixel_accuracy = self.evaluator.pixel_accuracy_class()
        mIoU = self.evaluator.mean_intersection_over_union()
        FWIoU = self.evaluator.frequency_weighted_intersection_over_union()
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        self.writer.add_scalar('train/pixel_accuracy', pixel_accuracy, epoch)
        self.writer.add_scalar('train/mean_class_pixel_accuracy', mean_class_pixel_accuracy, epoch)
        self.writer.add_scalar('train/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

        # update learning rate per epoch for step / cos mode
        if self.args.lr_scheduler != 'plateau':
            self.scheduler.step(epoch=epoch)

        print('train validation:')
        print("pixel_acc:{}, mean_class_pixel_acc:{}, mIoU:{}, fwIoU: {}".format(pixel_accuracy, mean_class_pixel_accuracy, mIoU, FWIoU))
        print('Loss: %.6f' % train_loss)
        print('---------------------------------')

    def validation(self, epoch):
        test_loss = 0.0
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        num_img_val = len(self.val_loader)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.6f' % (test_loss / (i + 1)))
            self.writer.add_scalar('val/total_loss_iter', loss.item(), i + num_img_val * epoch)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        pixel_accuracy = self.evaluator.pixel_accuracy()
        mean_class_pixel_accuracy = self.evaluator.pixel_accuracy_class()
        mIoU = self.evaluator.mean_intersection_over_union()
        FWIoU = self.evaluator.frequency_weighted_intersection_over_union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/pixel_accuracy', pixel_accuracy, epoch)
        self.writer.add_scalar('val/mean_class_pixel_accuracy', mean_class_pixel_accuracy, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('test validation:')
        print("pixel_acc:{}, mean_class_pixel_acc:{}, mIoU:{}, fwIoU: {}".format(pixel_accuracy, mean_class_pixel_accuracy, mIoU, FWIoU))
        print('Loss: %.6f' % test_loss)
        print('====================================')

        # update learning rate per epoch for plateau mode
        if self.args.lr_scheduler == 'plateau':
            self.scheduler.step(metrics=mIoU, epoch=epoch)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DSFNet Training")
    parser.add_argument('--backbone', type=str, default='light_xception',
                        choices=['xception', 'light_xception'], help='backbone name (default: light_xception)')
    parser.add_argument('--out-stride', type=int, default=8,
                        choices=[8, 16], help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['cityscapes'], help='dataset name (default: cityscapes)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--sync-bn', action='store_true', default=False,
                        help='whether to use sync bn (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'], help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--use-attention', action='store_true', default=False,
                        help='whether to use attention (default: False)')
    # optimizer params
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['sgd', 'adam'], help='Optimizer: (default: adam)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='plateau',
                        choices=['plateau', 'step', 'cos'], help='lr scheduler mode: (default: plateau)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        metavar='M', help='w-decay (default: 0.0)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {'cityscapes': 200}
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.lr is None:
        lrs = {'cityscapes': 0.001}
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'dsfnet-'+str(args.backbone)
    print(args)
    # 固定随机种子
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
