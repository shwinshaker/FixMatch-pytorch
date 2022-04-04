#!/bin/bash

# script='./train.py'
script='./train_mean_teacher.py'
# python -m torch.distributed.launch --nproc_per_node 8 $script --dataset cifar10 \
#                                                               --num-labeled 4000 \
#                                                               --arch wideresnet \
#                                                               --batch-size 8 \
#                                                               --lr 0.03 \
#                                                               --expand-labels \
#                                                               --seed 5 \
#                                                               --out results/cifar10@4000_mean_teacher_reinit_interpolate \
#                                                               --epochs 50 \
#                                                               --steps-per-epoch 1024 \
                                                              # --address 'tcp://127.0.0.1:23458'
                                                            #   --out results/cifar10@4000_mean_teacher

#  --amp --opt_level O2

python $script --dataset cifar10 \
               --num-labeled 4000 \
               --arch wideresnet \
               --batch-size 64 \
               --lr 0.03 \
               --expand-labels \
               --seed 5 \
               --out results/cifar10@4000_mean_teacher_1 \
               --epochs 50 \
               --steps-per-epoch 16 \
               --gpu-id 3 \
               --regularizer 'sparse' \
               --reinit_sparsity 0.3 \
              #  --reinit_retain_rate 0.99 \
#  --amp --opt_level O2