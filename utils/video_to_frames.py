# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:video_to_frames.py
"""
import os
import cv2
import argparse
import numpy as np


def video2frames(path, output_dir=None, skip=1, mirror=False):
    """
    Inputs:
    -------
    - path: 视频文件路径
    - output_dir: 图片保存目录
    - skip: 间隔多少帧保存一张图片
    - mirror: 是否水平翻转图片
    """
    video_object = cv2.VideoCapture(path)

    # 获得视频帧率、尺寸、帧总数
    fps = video_object.get(cv2.CAP_PROP_FPS)
    size = (int(video_object.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = video_object.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps: {}, size: {}, frame numbers: {}".format(fps, size, fNUMS))

    # setup the output folder
    if output_dir is None:
        file_path, file_name = os.path.split(path)  # 将视频路径分为 该级目录 与 文件名
        file_name, suffix = os.path.splitext(file_name)  # 将文件名分为 文件名 与文件后缀名
        output_dir = os.path.join(file_path, file_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    index = 0
    while video_object.isOpened():
        success, frame = video_object.read()
        if success:  # 是否读取成功
            if index % skip == 0:  # 间隔多少帧保存
                if mirror:  # 是否水平翻转图片
                    frame = np.fliplr(frame)
                cv2.imwrite("{}/{:05d}.jpg".format(output_dir, index), frame)
        else:
            raise IOError("Reading video file Error!")
        index += 1
    print('Finish!\nNumber of frames:' + str(index-1))


def main():
    parser = argparse.ArgumentParser("Enter the filename of a video")
    parser.add_argument('--filename', type=str, default=r'D:\EP01.mp4')
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('--skip', type=int, default=1, help="Only save every n th frame")
    parser.add_argument('--mirror', action='store_true', default=False, help="Flip every other image")
    args = parser.parse_args()

    # In case the filename points to a directory
    if os.path.isdir(args.filename):
        files = [os.path.join(args.filename, f) for f in os.listdir(args.filename) if os.path.isfile(os.path.join(args.filename, f))]
        for video in files:
            video2frames(video, output_dir=args.output_dir, skip=args.skip, mirror=args.mirror)
    else:
        video2frames(args.filename, output_dir=args.output_dir, skip=args.skip, mirror=args.mirror)


if __name__ == "__main__":
    main()
