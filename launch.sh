#!/bin/bash

# script='./train.py'
script='./train_mean_teacher.py'
gpus='2,3'
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node 2 $script --dataset cifar10 \
                                                              --num-labeled 4000 \
                                                              --arch wideresnet \
                                                              --batch-size 32 \
                                                              --lr 0.03 \
                                                              --expand-labels \
                                                              --seed 5 \
                                                              --out results/cifar10@4000_mean_teacher_reinit \
                                                              --address 'tcp://127.0.0.1:23463' \
                                                              --world-size 2 \
                                                              --epochs 1024 \
                                                              --steps-per-epoch 1024 \
                                                              --regularizer 'interpolate' \
                                                              --reinit_retain_rate 0.5 \

#  --amp --opt_level O2

# python $script --dataset cifar10 \
#                --num-labeled 4000 \
#                --arch wideresnet \
#                --batch-size 64 \
#                --lr 0.03 \
#                --expand-labels \
#                --seed 5 \
#                --out results/cifar10@4000_mean_teacher_reinit \
#                --epochs 50 \
#                --steps-per-epoch 16 \
#                --gpu-id 3 \
#                --regularizer 'interpolate' \
#                --reinit_retain_rate 0.99 \
              #  --regularizer 'sparse' \
              #  --reinit_sparsity 0.3 \
#  --amp --opt_level O2