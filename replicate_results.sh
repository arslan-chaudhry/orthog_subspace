# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#! /bin/bash
# Usage ./replicate_results.sh
set -e

NUM_RUNS=5
BATCH_SIZE=10
EPS_MEM_BATCH_SIZE=10
MEM_SIZE=1
LOG_DIR='results/'

if [ ! -d $LOG_DIR ]; then
    mkdir -pv $LOG_DIR
fi

#####################################################
############## MNIST Permutations ##################
####################################################
EXAMPLES_PER_TASK=10000
# Finetune
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method VAN --synap-stgth 0.0 --log-dir $LOG_DIR --examples-per-task $EXAMPLES_PER_TASK
# EWC
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method EWC --synap-stgth 10.0 --log-dir $LOG_DIR --examples-per-task $EXAMPLES_PER_TASK
# A-GEM
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method A-GEM --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size $MEM_SIZE --eps-mem-batch $EPS_MEM_BATCH_SIZE --examples-per-task $EXAMPLES_PER_TASK
# MER
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03 --imp-method MER --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size $MEM_SIZE --eps-mem-batch $EPS_MEM_BATCH_SIZE --examples-per-task $EXAMPLES_PER_TASK
# ER-Ringbuffer
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method ER-Ringbuffer --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size $MEM_SIZE --eps-mem-batch $EPS_MEM_BATCH_SIZE --examples-per-task $EXAMPLES_PER_TASK
# Orthog Subspace w/o memory (Ours)
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method SUBSPACE-PROJ --synap-stgth 0.0 --log-dir $LOG_DIR --examples-per-task $EXAMPLES_PER_TASK


#####################################################
############## MNIST Rotations ##################
####################################################
EXAMPLES_PER_TASK=10000
# Finetune
python fc_rotate_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method VAN --synap-stgth 0.0 --log-dir $LOG_DIR --examples-per-task $EXAMPLES_PER_TASK
# EWC
python fc_rotate_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method EWC --synap-stgth 10.0 --log-dir $LOG_DIR --examples-per-task $EXAMPLES_PER_TASK
# A-GEM
python fc_rotate_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method A-GEM --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size $MEM_SIZE --eps-mem-batch $EPS_MEM_BATCH_SIZE --examples-per-task $EXAMPLES_PER_TASK
# MER
python fc_rotate_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03 --imp-method MER --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size $MEM_SIZE --eps-mem-batch $EPS_MEM_BATCH_SIZE --examples-per-task $EXAMPLES_PER_TASK
# ER-Ringbuffer
python fc_rotate_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method ER-Ringbuffer --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size $MEM_SIZE --eps-mem-batch $EPS_MEM_BATCH_SIZE --examples-per-task $EXAMPLES_PER_TASK
# Orthog Subspace w/o memory (Ours)
python fc_rotate_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1 --imp-method SUBSPACE-PROJ --synap-stgth 0.0 --log-dir $LOG_DIR --examples-per-task $EXAMPLES_PER_TASK


#####################################################
############## Split CIFAR ##################
####################################################
# Finetune
python conv_split_cifar.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method VAN --synap-stgth 0.0 --log-dir $LOG_DIR
# EWC
python conv_split_cifar.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method EWC --synap-stgth 10.0 --log-dir $LOG_DIR
# A-GEM
python conv_split_cifar.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method A-GEM --synap-stgth 0.0 --log-dir $LOG_DIR
# ER-Ringbuffer
python conv_split_cifar.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method ER-Ringbuffer --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size 1 --eps-mem-batch $EPS_MEM_BATCH_SIZE
# Orthog Subspace with memory (Ours)
python conv_split_cifar.py --maintain-orthogonality --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.4 --imp-method ER-SUBSPACE --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size 1 --eps-mem-batch $EPS_MEM_BATCH_SIZE


#####################################################
############## Split miniImageNet ##################
####################################################
if [ ! -f ./miniImageNet_Dataset/miniImageNet_full.pickle ]; then
    echo "miniImageNet dataset not found! Please download the dataset."
    exit -1
fi
# Finetune
python conv_split_miniImagenet.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method VAN --synap-stgth 0.0 --log-dir $LOG_DIR
# EWC
python conv_split_miniImagenet.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method EWC --synap-stgth 10.0 --log-dir $LOG_DIR
# A-GEM
python conv_split_miniImagenet.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method A-GEM --synap-stgth 0.0 --log-dir $LOG_DIR
# ER-Ringbuffer
python conv_split_miniImagenet.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method ER-Ringbuffer --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size 1 --eps-mem-batch $EPS_MEM_BATCH_SIZE
# Orthog Subspace with memory (Ours)
python conv_split_miniImagenet.py --maintain-orthogonality --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --num-tasks 20 --batch-size $BATCH_SIZE --learning-rate 0.2 --imp-method ER-SUBSPACE --synap-stgth 0.0 --log-dir $LOG_DIR --mem-size 1 --eps-mem-batch $EPS_MEM_BATCH_SIZE
