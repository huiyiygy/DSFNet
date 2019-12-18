#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python vis_eval.py --backbone light_xception --use-attention --use-attention --crop-size 1024 --out-stride 8 --batch-size 10 --is-vis --checkpoint-folder /home/lab/ygy/DSFNet/checkpoint/cityscapes/saved/019_60.50_attention_os8_lr0.01_delay_reduction_channels_numbers_in_attention