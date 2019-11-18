#!/usr/bin/env bash

# experiment 0
CUDA_VISIBLE_DEVICES=0,1 python train.py --lr 0.001 --weight-decay 0.0 --sync-bn --is-native --epochs 200 --batch-size 12 --base-size 513 --crop-size 513 --gpu-ids 0,1 --checkname dsfnet --eval-interval 1