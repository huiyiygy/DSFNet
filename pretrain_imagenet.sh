#!/usr/bin/env bash

# experiment 1 with channel shuffle
# CUDA_VISIBLE_DEVICES=0 python pretrain_imagenet.py --epochs 90 -b 40 --lr 0.1 --gpu 0 --use-channel-shuffle

# experiment 2 without channel shuffle
CUDA_VISIBLE_DEVICES=0 python pretrain_imagenet.py --epochs 90 -b 50 --lr 0.1 --gpu 0
