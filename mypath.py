# -*- coding: utf-8 -*-
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return '/home/lab/ygy/Cityscapes/'     # folder that contains leftImg8bit/
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
