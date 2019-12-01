#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python pretrain_imagenet.py --epochs 90 -b 40 --lr 0.1 --gpu 0 -p 50
