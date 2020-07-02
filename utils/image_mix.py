# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:image_mix.py
@time:2020/07/02 20:24
"""
import os
import cv2
import numpy as np
from tqdm import tqdm


def main():
    rate = 0.4
    img_dir_A = r'D:\15_video_frames'
    img_dir_B = r'D:\15_video_frames_predict'
    output_dir = r'D:\15_video_frames_mix'

    filenames_A = os.listdir(img_dir_A)
    filenames_A = sorted(filenames_A)
    filenamex_A = [os.path.join(img_dir_A, x) for x in filenames_A]

    filenames_B = os.listdir(img_dir_B)
    filenames_B = sorted(filenames_B)
    filenamex_B = [os.path.join(img_dir_B, x) for x in filenames_B]

    for i in range(0, len(filenamex_A)):
        img_A = cv2.imread(filenamex_A[i])
        img_B = cv2.imread(filenamex_B[i])

        img = img_A * rate + img_B * (1-rate)
        _, file_name = os.path.split(filenamex_A[i])  # 将路径分为 该级目录 与 文件名
        cv2.imwrite("{}/{}".format(output_dir, file_name), img)

    print('Finished!')


if __name__ == '__main__':
    main()
