#!/usr/bin/env bash

# experiment 0
CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --weight-decay 0.0 --is-native --epochs 500 --batch-size 8 --base-size 513 --crop-size 513 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 1 use-balanced-weights, half channel
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --weight-decay 0.0 --is-native --use-balanced-weights --epochs 500 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1
