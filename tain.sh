#!/usr/bin/env bash

# experiment 0 native DSFNet out-stride 8
CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --out-stride 8 --epochs 400 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 1 native DSFNet use-balanced-weights out-stride 8
CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-balanced-weights --out-stride 8 --epochs 400 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 2 native DSFNet out-stride 16
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --out-stride 16 --epochs 400 --batch-size 16 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 3 native DSFNet use-balanced-weights out-stride 16
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-balanced-weights --out-stride 16 --epochs 400 --batch-size 16 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

