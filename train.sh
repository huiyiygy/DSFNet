#!/usr/bin/env bash

# experiment 1 native DSFNet out-stride 8
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 2 native DSFNet use-balanced-weights out-stride 8
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-balanced-weights --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 3 native DSFNet out-stride 16
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --out-stride 16 --epochs 300 --batch-size 16 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 4 native DSFNet use-balanced-weights out-stride 16
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-balanced-weights --out-stride 16 --epochs 300 --batch-size 16 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 5 attention DSFNet out-stride 8
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 6 attention DSFNet use-balanced-weights out-stride 8 focal loss
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.001 --use-balanced-weights --use-attention --out-stride 8 --loss-type focal --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 7 attention DSFNet out-stride 16 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 16 --epochs 300 --batch-size 16 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 8 attention DSFNet  out-stride 16 lr 0.01 nesterov sgd
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --optim sgd --nesterov --use-attention --out-stride 16 --epochs 300 --batch-size 16 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 9 attention DSFNet out-stride 16 lr 0.1
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.1 --use-attention --out-stride 16 --epochs 300 --batch-size 16 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 10 attention DSFNet out-stride 8 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 11 attention DSFNet out-stride 8 lr 0.1
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.1 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 12 attention DSFNet out-stride 8 lr 0.01 weight decay 1e-5
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --weight-decay 1e-5 --use-attention --out-stride 8 --epochs 400 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 13 attention DSFNet out-stride 8 lr 0.01 new attention mix method
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 14 attention DSFNet out-stride 8 lr 0.01 new attention mix method , decoder no DSFConv
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 15 native DSFNet out-stride 8 lr 0.01
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 16 attention DSFNet out-stride 8 lr 0.01 new attention mix method , attention no sigmoid
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 17 attention DSFNet out-stride 8 lr 0.01 new attention
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 18 attention DSFNet out-stride 8 lr 0.01 new attention spatial branch channels change to 1
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 19 attention DSFNet out-stride 8 lr 0.01 new attention delay reduction channels numbers in the attention branch
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 20 attention DSFNet out-stride 8 lr 0.01 new attention delay reduction channels numbers in the attention branch No channel attention
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1

# experiment 21 attention DSFNet out-stride 8 lr 0.01 new attention delay reduction channels numbers in the attention branch No spatial attention
CUDA_VISIBLE_DEVICES=0 python train.py --lr 0.01 --use-attention --out-stride 8 --epochs 300 --batch-size 8 --base-size 512 --crop-size 512 --gpu-ids 0 --checkname dsfnet --eval-interval 1