#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python vis_eval.py --crop-size 512 --out-stride 8 --batch-size 20 --use-attention --is-vis --checkpoint-folder /home/lab/ygy/DSFNet/checkpoint/cityscapes/dsfnet/experiment_0/