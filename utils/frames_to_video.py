# -*- coding:utf-8 -*-
"""
@function:
@author:HuiYi or 会意
@file:frames_to_video.py
"""
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import argparse
from tqdm import tqdm


def write(images, fps=25, size=None, is_color=True, format="mp4V", video_name='demo.mp4'):
    """
    Inputs:
    -------
    - images: 存放视频帧的目录
    - fps: 视频帧率
    - size: 视频分辨率大小
    - is_color: 是否保存为彩色
    - format: 'mp4V' for mp4, 'XVID' for avi
    - video_name: 视频保存路径
    """
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    tbar = tqdm(images)
    for image in tbar:
        if image.split('.')[-1] != 'jpg':
            continue
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(video_name, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def jpg2video(images_dir, out_dir, fps, out_size):
    filenames = os.listdir(images_dir)
    filenames = sorted(filenames)
    filenamex = [os.path.join(images_dir, x) for x in filenames]
    video_name = images_dir.split('/')[-1] + '.mp4'
    if out_dir is None:
        out_dir = os.path.dirname(images_dir)
    write(filenamex, fps=fps, size=out_size, video_name=os.path.join(out_dir, video_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default=r'D:\EP01_images')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--out_width', type=int, default=1920)
    parser.add_argument('--out_height', type=int, default=1080)
    args = parser.parse_args()
    jpg2video(args.images_dir, args.out_dir, args.fps, (args.out_width, args.out_height))
    print('Finish!')


if __name__ == '__main__':
    main()