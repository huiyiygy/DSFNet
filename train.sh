#!/usr/bin/env bash

# experiment 1 native DSFNet out-stride 8
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --out-stride 8

# experiment 2 native DSFNet use-balanced-weights out-stride 8
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-balanced-weights --out-stride 8

# experiment 3 native DSFNet out-stride 16
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --out-stride 16 --batch-size 16

# experiment 4 native DSFNet use-balanced-weights out-stride 16
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-balanced-weights --out-stride 16 --batch-size 16

# experiment 5 attention DSFNet out-stride 8
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-attention --use-channel-shuffle --out-stride 8

# experiment 6 attention DSFNet use-balanced-weights out-stride 8 focal loss
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-balanced-weights --use-attention --use-channel-shuffle --out-stride 8 --loss-type focal

# experiment 7 attention DSFNet out-stride 16 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 16 --batch-size 16

# experiment 8 attention DSFNet  out-stride 16 lr 0.01 nesterov sgd
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --optim sgd --nesterov --use-attention --use-channel-shuffle --out-stride 16--batch-size 16

# experiment 9 attention DSFNet out-stride 16 lr 0.1
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.1 --use-attention --use-channel-shuffle --out-stride 16 --batch-size 16

# experiment 10 attention DSFNet out-stride 8 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 11 attention DSFNet out-stride 8 lr 0.1
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.1 --use-attention --use-channel-shuffle --out-stride 8

# experiment 12 attention DSFNet out-stride 8 lr 0.01 weight decay 1e-5
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --weight-decay 1e-5 --use-attention --use-channel-shuffle --out-stride 8 --epochs 400

# experiment 13 attention DSFNet out-stride 8 lr 0.01 new attention mix method
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 14 attention DSFNet out-stride 8 lr 0.01 new attention mix method , decoder no DSFConv
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 15 native DSFNet out-stride 8 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --out-stride 8

# experiment 16 attention DSFNet out-stride 8 lr 0.01 new attention mix method , attention no sigmoid
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 17 attention DSFNet out-stride 8 lr 0.01 new attention
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 18 attention DSFNet out-stride 8 lr 0.01 new attention spatial branch channels change to 1
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 19 attention DSFNet out-stride 8 lr 0.01 new attention delay reduction channels numbers in the attention branch
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 20 attention DSFNet out-stride 8 lr 0.01 new attention delay reduction channels numbers in the attention branch No channel attention
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 21 attention DSFNet out-stride 8 lr 0.01 new attention delay reduction channels numbers in the attention branch No spatial attention
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --use-channel-shuffle --out-stride 8

# experiment 22 attention SeparableXception out-stride 8 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone separable_xception --workers 16 --lr 0.01 --use-attention --out-stride 8

# experiment 23 attention SeparableXception out-stride 8 lr 0.01 decoder use separable conv
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone separable_xception --workers 16 --lr 0.01 --use-attention --out-stride 8

# experiment 24 native SeparableXception out-stride 8 lr 0.01 decoder use separable conv
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone separable_xception --workers 16 --lr 0.01 --out-stride 8

# experiment 25 attention DSFNet out-stride 8 lr 0.01 without channel shuffle
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8

# experiment 26 attention SeparableXception out-stride 8 lr 0.01 decoder use separable conv No channel attention
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone separable_xception --lr 0.01 --use-attention --out-stride 8

# experiment 27 attention SeparableXception out-stride 8 lr 0.01 decoder use separable conv No spatial attention
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone separable_xception --lr 0.01 --use-attention --out-stride 8

# experiment 28 native DSFNet out-stride 8 lr 0.01 without channel shuffle
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --out-stride 8

# experiment 29 attention DSFNet out-stride 8 lr 0.01 without channel shuffle without spatial attention
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8

# experiment 30 attention DSFNet out-stride 8 lr 0.01 without channel shuffle without channel attention
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8

# experiment 31 attention NativeXception out-stride 8 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone native_xception --epochs 400 --lr 0.01 --use-attention --out-stride 8

# experiment 32 native NativeXception out-stride 8 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone native_xception --lr 0.01 --out-stride 8

# experiment 33 attention NativeXception out-stride 8 lr 0.01 without spatial attention
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone native_xception --lr 0.01 --use-attention --out-stride 8

# experiment 34 attention NativeXception out-stride 8 lr 0.01 without channel attention
# CUDA_VISIBLE_DEVICES=0 python train.py --backbone native_xception --lr 0.01 --use-attention --out-stride 8

# experiment 35 pretrained attention DSFNet out-stride 8 lr 0.001 without channel shuffle
CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-attention --out-stride 8 --pretrained --pretrain-file /home/lab/ygy/DSFNet/checkpoint/imagenet/02_59.002_without_ channel_ shuffle/pretrain_model_best.pth.tar

# experiment 36 pretrained attention DSFNet out-stride 8 lr 0.001 with channel shuffle
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-attention --use-channel-shuffle --out-stride 8 --pretrained --pretrain-file /home/lab/ygy/DSFNet/checkpoint/imagenet/01_54.468_with_ channel_ shuffle/pretrain_model_best.pth.tar