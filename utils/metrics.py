# -*- coding: utf-8 -*-
import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def pixel_accuracy(self):
        """
        标记正确的像素占总像素的比例
        """
        PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return PA

    def pixel_accuracy_class(self):
        pac = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mpac = np.nanmean(pac)  # 所有求平均
        return mpac

    def mean_intersection_over_union(self):
        per_class_iou = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(per_class_iou)
        return [per_class_iou, MIoU]

    def frequency_weighted_intersection_over_union(self):
        """
        MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
