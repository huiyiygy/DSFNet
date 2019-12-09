#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python vis_eval.py --crop-size 1024 --out-stride 8 --batch-size 10 --is-vis --checkpoint-folder /home/lab/ygy/DSFNet/checkpoint/cityscapes/dsfnet/015_59.23_native_os8_lr0.01/