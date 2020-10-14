# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for split MNIST experiment.
"""
from __future__ import print_function

import argparse
import os
import sys
import math
import time

import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from six.moves import cPickle as pickle

from utils.data_utils import construct_split_mnist
from utils.utils import get_sample_weights, sample_from_dataset, update_episodic_memory, concatenate_datasets, samples_for_each_class, sample_from_dataset_icarl, compute_fgt, update_reservior, update_fifo_buffer, average_acc_stats_across_runs, average_fgt_stats_across_runs, generate_projection_matrix, unit_test_projection_matrices, grad_similarity_check, grad_check, load_task_specific_data
from utils.vis_utils import plot_acc_multiple_runs, plot_histogram, snapshot_experiment_meta_data, snapshot_experiment_eval
from model import Model

###############################################################
################ Some definitions #############################
### These will be edited by the command line options ##########
###############################################################

## Training Options
NUM_RUNS = 10           # Number of experiments to average over
TRAIN_ITERS = 5000      # Number of training iterations per task
BATCH_SIZE = 10
LEARNING_RATE = 1e-3    
RANDOM_SEED = 1234
VALID_OPTIMS = ['SGD', 'MOMENTUM', 'ADAM']
OPTIM = 'SGD'
OPT_POWER = 0.9
OPT_MOMENTUM = 0.9
VALID_ARCHS = ['FC-S', 'FC-B']
ARCH = 'FC-S'

## Model options
MODELS = ['VAN', 'PI', 'EWC', 'MAS', 'RWALK', 'A-GEM', 'S-GEM', 'FTR_EXT', 'PNN', 'ER-Reservoir', 'ER-Ringbuffer', 'SUBSPACE-PROJ', 'ER-SUBSPACE', 'ER-PROJ-SUBSPACE', 'PROJ-ANCHOR', 'PROJ-SUBSPACE-GRADIENT-PENALTY'] #List of valid models 
IMP_METHOD = 'EWC'
SYNAP_STGTH = 75000
FISHER_EMA_DECAY = 0.9      # Exponential moving average decay factor for Fisher computation (online Fisher)
FISHER_UPDATE_AFTER = 10    # Number of training iterations for which the F_{\theta}^t is computed (see Eq. 10 in RWalk paper) 
SAMPLES_PER_CLASS = 25   # Number of samples per task
INPUT_FEATURE_SIZE = 784
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNELS = 1
TOTAL_CLASSES = 10          # Total number of classes in the dataset 
EPS_MEM_BATCH_SIZE = 256
DEBUG_EPISODIC_MEMORY = False
USE_GPU = True
K_FOR_CROSS_VAL = 0
TIME_MY_METHOD = False
COUNT_VIOLATIONS = False
MEASURE_PERF_ON_EPS_MEMORY = False

## Logging, saving and testing options
LOG_DIR = './split_mnist_results'

## Evaluation options

## Num Tasks
NUM_TASKS = 23
MULTI_TASK = False
PROJECTION_RANK = 12
GRAD_CHECK = False
QR= False

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for permutted mnist experiment.")
    parser.add_argument("--cross-validate-mode", action="store_true",
                       help="If option is chosen then snapshoting after each batch is disabled")
    parser.add_argument("--online-cross-val", action="store_true",
                       help="If option is chosen then enable the online cross validation of the learning rate")
    parser.add_argument("--train-single-epoch", action="store_true",
                       help="If option is chosen then train for single epoch")
    parser.add_argument("--eval-single-head", action="store_true",
                       help="If option is chosen then evaluate on a single head setting.")
    parser.add_argument("--arch", type=str, default=ARCH, help="Network Architecture for the experiment.\
                        \n \nSupported values: %s"%(VALID_ARCHS))
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS,
                       help="Total runs/ experiments over which accuracy is averaged.")
    parser.add_argument("--num-tasks", type=int, default=NUM_TASKS,
                       help="Total number of tasks.")
    parser.add_argument("--train-iters", type=int, default=TRAIN_ITERS,
                       help="Number of training iterations for each task.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Mini-batch size for each task.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random Seed.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                       help="Starting Learning rate for each task.")
    parser.add_argument("--optim", type=str, default=OPTIM,
                        help="Optimizer for the experiment. \
                                \n \nSupported values: %s"%(VALID_OPTIMS))
    parser.add_argument("--imp-method", type=str, default=IMP_METHOD,
                       help="Model to be used for LLL. \
                        \n \nSupported values: %s"%(MODELS))
    parser.add_argument("--synap-stgth", type=float, default=SYNAP_STGTH,
                       help="Synaptic strength for the regularization.")
    parser.add_argument("--fisher-ema-decay", type=float, default=FISHER_EMA_DECAY,
                       help="Exponential moving average decay for Fisher calculation at each step.")
    parser.add_argument("--fisher-update-after", type=int, default=FISHER_UPDATE_AFTER,
                       help="Number of training iterations after which the Fisher will be updated.")
    parser.add_argument("--mem-size", type=int, default=SAMPLES_PER_CLASS,
                       help="Number of samples per class from previous tasks.")
    parser.add_argument("--eps-mem-batch", type=int, default=EPS_MEM_BATCH_SIZE,
                       help="Number of samples per class from previous tasks.")
    parser.add_argument("--examples-per-task", type=int, default=1000,
                       help="Number of examples per task.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                       help="Directory where the plots and model accuracies will be stored.")
    return parser.parse_args()

def train_task_sequence(model, sess, datasets, args):
    """
    Train and evaluate LLL system such that we only see a example once
    Args:
    Returns:
        dict    A dictionary containing mean and stds for the experiment
    """
    # List to store accuracy for each run
    runs = []

    batch_size = args.batch_size

    if model.imp_method == 'A-GEM' or 'ER-' in model.imp_method:
        use_episodic_memory = True
    else:
        use_episodic_memory = False

    # Loop over number of runs to average over
    for runid in range(args.num_runs):
        print('\t\tRun %d:'%(runid))

        # Initialize the random seeds
        np.random.seed(args.random_seed+runid)

        # Get the task labels from the total number of tasks and full label space
        task_labels = []
        classes_per_task = TOTAL_CLASSES// args.num_tasks
        total_classes = classes_per_task * model.num_tasks
        if args.online_cross_val:
            label_array = np.arange(total_classes)
        else:
            class_label_offset = K_FOR_CROSS_VAL * classes_per_task
            label_array = np.arange(class_label_offset, total_classes+class_label_offset)

        np.random.shuffle(label_array)
        for tt in range(model.num_tasks):
            tt_offset = tt*classes_per_task
            task_labels.append(list(label_array[tt_offset:tt_offset+classes_per_task]))
            print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))

        episodic_mem_size = args.mem_size * TOTAL_CLASSES

        # Initialize all the variables in the model
        sess.run(tf.global_variables_initializer())

        # Run the init ops
        model.init_updates(sess)

        # List to store accuracies for a run
        evals = []

        # List to store the classes that we have so far - used at test time
        test_labels = []

        if use_episodic_memory:
            # Reserve a space for episodic memory
            episodic_images = np.zeros([episodic_mem_size, INPUT_FEATURE_SIZE])
            episodic_labels = np.zeros([episodic_mem_size, TOTAL_CLASSES])
            count_cls = np.zeros(TOTAL_CLASSES, dtype=np.int32)
            episodic_filled_counter = 0
            examples_seen_so_far = 0

        # Mask for softmax
        # Since all the classes are present in all the tasks so nothing to mask
        logit_mask = np.zeros(TOTAL_CLASSES)

        if model.imp_method == 'PNN':
            pnn_train_phase = np.array(np.zeros(model.num_tasks), dtype=np.bool)
            pnn_logit_mask = np.ones([model.num_tasks, TOTAL_CLASSES])

        if COUNT_VIOLATIONS:
            violation_count = np.zeros(model.num_tasks)
            vc = 0

        # Store the projection matrices for each task
        proj_matrices = generate_projection_matrix(model.num_tasks, feature_dim=model.subspace_proj.get_shape()[0], qr=QR)
        # Check the sanity of the generated matrices
        unit_test_projection_matrices(proj_matrices)

        # TODO: Temp for gradients check
        prev_task_grads = []
        # Training loop for all the tasks
        for task in range(len(task_labels)):
            print('\t\tTask %d:'%(task))

            # If not the first task then restore weights from previous task
            if(task > 0 and model.imp_method != 'PNN'):
                model.restore(sess)

            task_train_images, task_train_labels = load_task_specific_data(datasets[0]['train'], task_labels[task])

            if MULTI_TASK:
                if task == 0:
                    for t_ in range(1, len(task_labels)):
                        task_tr_images, task_tr_labels = load_task_specific_data(datasets[0]['train'], task_labels[t_])
                        task_train_images = np.concatenate((task_train_images, task_tr_images), axis=0)
                        task_train_labels = np.concatenate((task_train_labels, task_tr_labels), axis=0)
                else:
                    # Skip training for this task
                    continue

            print('Received {} images, {} labels at task {}'.format(task_train_images.shape[0], task_train_labels.shape[0], task))
            print('Unique labels in the task: {}'.format(np.unique(np.nonzero(task_train_labels)[1])))

            # Array to store accuracies when training for task T
            ftask = []
           
            # Assign equal weights to all the examples
            task_sample_weights = np.ones([task_train_labels.shape[0]], dtype=np.float32)

            num_train_examples = task_train_images.shape[0]

            
            # Train a task observing sequence of data
            logit_mask[:] = 0
            if args.train_single_epoch:
                num_iters = (num_train_examples + batch_size - 1) // batch_size
                if args.cross_validate_mode:
                    logit_mask[task_labels[task]] = 1.0
            else:
                num_iters = args.train_iters
                logit_mask[task_labels[task]] = 1.0

            # Randomly suffle the training examples
            perm = np.arange(num_train_examples)
            np.random.shuffle(perm)
            train_x = task_train_images[perm]
            train_y = task_train_labels[perm]
            task_sample_weights = task_sample_weights[perm]
    
            # Training loop for task T
            for iters in range(num_iters):

                if args.train_single_epoch and not args.cross_validate_mode:
                    if (iters < 10) or (iters < 100 and iters % 10 == 0) or (iters % 100 == 0):
                        # Snapshot the current performance across all tasks after each mini-batch
                        fbatch = test_task_sequence(model, sess, datasets[0]['test'], task_labels, task, args.online_cross_val, proj_matrices)
                        ftask.append(fbatch)

                offset = (iters * batch_size) % num_train_examples
                if (offset+batch_size <= num_train_examples):
                    residual = batch_size
                else:
                    residual = num_train_examples - offset

                if model.imp_method == 'PNN':
                    pnn_train_phase[:] = False
                    pnn_train_phase[task] = True
                    feed_dict = {model.x: train_x[offset:offset+batch_size], model.y_[task]: train_y[offset:offset+batch_size], 
                            model.sample_weights: task_sample_weights[offset:offset+batch_size],
                                 model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0, model.learning_rate: args.learning_rate}
                    train_phase_dict = {m_t: i_t for (m_t, i_t) in zip(model.train_phase, pnn_train_phase)}
                    logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, pnn_logit_mask)}
                    feed_dict.update(train_phase_dict)
                    feed_dict.update(logit_mask_dict)
                else:
                    feed_dict = {model.x: train_x[offset:offset+batch_size], model.y_: train_y[offset:offset+batch_size], 
                            model.sample_weights: task_sample_weights[offset:offset+batch_size],
                            model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0, 
                                 model.output_mask: logit_mask, model.train_phase: True, model.learning_rate: args.learning_rate}

                if model.imp_method == 'VAN':
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)


                elif model.imp_method == 'PNN':
                    feed_dict[model.task_id] = task
                    _, loss = sess.run([model.train[task], model.unweighted_entropy[task]], feed_dict=feed_dict)

                elif model.imp_method == 'FTR_EXT':
                    if task == 0:
                        _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
                    else:
                        _, loss = sess.run([model.train_classifier, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'EWC':
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                    # Update fisher after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        sess.run(model.set_running_fisher)
                        sess.run(model.reset_tmp_fisher)
                    
                    _, _, loss = sess.run([model.set_tmp_fisher, model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'PI':
                    _, _, _, loss = sess.run([model.weights_old_ops_grouped, model.train, model.update_small_omega, 
                                              model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'MAS':
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
               
                elif model.imp_method == 'PROJ-ANCHOR':
                    if task == 0:
                        feed_dict[model.subspace_proj] = proj_matrices[task]
                        _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
                        reg = 0.0     
                    else:
                        # Store the gradients in the orthogonal compliment
                        feed_dict[model.subspace_proj] = np.eye(proj_matrices[task].shape[0]) - proj_matrices[task]
                        _, reg = sess.run([model.store_ref_grads, model.anchor_loss], feed_dict=feed_dict)
                        feed_dict[model.subspace_proj] = proj_matrices[task]
                        _, loss = sess.run([model.train_subspace_proj, model.reg_loss], feed_dict=feed_dict)
                
                elif model.imp_method == 'PROJ-SUBSPACE-GRADIENT-PENALTY':
                    if task == 0:
                        feed_dict[model.subspace_proj] = proj_matrices[task]
                        _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
                        reg = 0.0     
                    else:
                        # Store the gradients in the orthogonal compliment
                        feed_dict[model.subspace_proj] = np.eye(proj_matrices[task].shape[0]) - proj_matrices[task]
                        sess.run(model.store_ref_grads, feed_dict=feed_dict)
                        feed_dict[model.subspace_proj] = proj_matrices[task]
                        _, loss = sess.run([model.train_gp, model.gp_total_loss], feed_dict=feed_dict)
                    
                elif model.imp_method == 'SUBSPACE-PROJ':
                    feed_dict[model.subspace_proj] = proj_matrices[task]
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
                    
                elif model.imp_method == 'A-GEM':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch, replace=False) # Sample without replacement so that we don't sample an example more than once
                        # Store the reference gradient
                        sess.run(model.store_ref_grads, feed_dict={model.x: episodic_images[mem_sample_mask], model.y_: episodic_labels[mem_sample_mask],
                                    model.keep_prob: 1.0, model.output_mask: logit_mask, model.train_phase: True, model.learning_rate: args.learning_rate})
                        if COUNT_VIOLATIONS:
                            vc, _, loss = sess.run([model.violation_count, model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)
                        else:
                            # Compute the gradient for current task and project if need be
                            _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)
                    # Put the batch in the ring buffer
                    update_fifo_buffer(train_x[offset:offset+residual], train_y[offset:offset+residual], episodic_images, episodic_labels,
                                       task_labels[task], args.mem_size, count_cls, episodic_filled_counter)

                elif model.imp_method == 'RWALK':
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                        # Store the current value of the weights
                        sess.run(model.weights_delta_old_grouped)
                    # Update fisher and importance score after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        # Update the importance score using distance in riemannian manifold   
                        sess.run(model.update_big_omega_riemann)
                        # Now that the score is updated, compute the new value for running Fisher
                        sess.run(model.set_running_fisher)
                        # Store the current value of the weights
                        sess.run(model.weights_delta_old_grouped)
                        # Reset the delta_L
                        sess.run([model.reset_small_omega])

                    _, _, _, _, loss = sess.run([model.set_tmp_fisher, model.weights_old_ops_grouped, 
                        model.train, model.update_small_omega, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'ER-Reservoir':
                    mem_filled_so_far = examples_seen_so_far if (examples_seen_so_far < episodic_mem_size) else episodic_mem_size
                    if mem_filled_so_far < args.eps_mem_batch:
                        er_mem_indices = np.arange(mem_filled_so_far)
                    else:
                        er_mem_indices = np.random.choice(mem_filled_so_far, args.eps_mem_batch, replace=False)
                    np.random.shuffle(er_mem_indices)
                    # Train on a batch of episodic memory first
                    er_train_x_batch = np.concatenate((episodic_images[er_mem_indices], train_x[offset:offset+residual]), axis=0)
                    er_train_y_batch = np.concatenate((episodic_labels[er_mem_indices], train_y[offset:offset+residual]), axis=0)
                    logit_mask[:] = 0
                    for tt in range(task+1):
                        logit_mask[task_labels[tt]] = 1.0
                    feed_dict = {model.x: er_train_x_batch, model.y_: er_train_y_batch,
                        model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0,
                                 model.output_mask: logit_mask, model.train_phase: True, model.learning_rate: args.learning_rate}
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
                    for er_x, er_y_ in zip(train_x[offset:offset+residual], train_y[offset:offset+residual]):
                        update_reservior(er_x, er_y_, episodic_images, episodic_labels, episodic_mem_size, examples_seen_so_far)
                        examples_seen_so_far += 1

                elif model.imp_method == 'ER-Ringbuffer':
                    mem_filled_so_far = episodic_filled_counter if (episodic_filled_counter <= episodic_mem_size) else episodic_mem_size
                    er_mem_indices = np.arange(mem_filled_so_far) if (mem_filled_so_far <= args.eps_mem_batch) else np.random.choice(mem_filled_so_far, 
                                                                                                                                     args.eps_mem_batch, replace=False)
                    er_train_x_batch = np.concatenate((episodic_images[er_mem_indices], train_x[offset:offset+residual]), axis=0)
                    er_train_y_batch = np.concatenate((episodic_labels[er_mem_indices], train_y[offset:offset+residual]), axis=0)
                    logit_mask[:] = 0
                    for tt in range(task+1):
                        logit_mask[task_labels[tt]] = 1.0
                    feed_dict = {model.x: er_train_x_batch, model.y_: er_train_y_batch,
                                 model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0,
                                 model.output_mask: logit_mask, model.learning_rate: args.learning_rate}
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                    # Put the batch in the FIFO ring buffer
                    update_fifo_buffer(train_x[offset:offset+residual], train_y[offset:offset+residual], episodic_images, episodic_labels,
                                       task_labels[task], args.mem_size, count_cls, episodic_filled_counter)
               
                elif model.imp_method == 'ER-PROJ-SUBSPACE':
                    if task == 0:
                        # Do normal training
                        feed_dict[model.subspace_proj] = np.eye(proj_matrices[task].shape[0]) - proj_matrices[task]
                        sess.run([model.store_ref_grads], feed_dict=feed_dict)
                        feed_dict[model.subspace_proj] = proj_matrices[task]
                        _, reg, loss = sess.run([model.train_subspace_proj, model.regularization, model.reg_loss], feed_dict=feed_dict)
                    else:
                        # Compute the gradients using memory in the orthogonal compliment space
                        mem_filled_so_far = episodic_filled_counter if (episodic_filled_counter <= episodic_mem_size) else episodic_mem_size
                        er_mem_indices = np.arange(mem_filled_so_far) if (mem_filled_so_far <= args.eps_mem_batch) else np.random.choice(mem_filled_so_far, 
                                                                                                                                args.eps_mem_batch, replace=False)
                        feed_dict = {model.x: episodic_images[er_mem_indices], model.y_: episodic_labels[er_mem_indices],
                                     model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0,
                                     model.output_mask: logit_mask, model.learning_rate: args.learning_rate}
                        feed_dict[model.subspace_proj] = np.eye(proj_matrices[task].shape[0]) - proj_matrices[task]
                        sess.run([model.store_ref_grads], feed_dict=feed_dict)

                        # Train on the current task and subtract any projection in the orthogonal compliment space
                        feed_dict[model.x] = train_x[offset:offset+residual] 
                        feed_dict[model.y_] = train_y[offset:offset+residual]
                        feed_dict[model.subspace_proj] = proj_matrices[task]
                        _, reg, loss = sess.run([model.train_subspace_proj, model.regularization, model.reg_loss], feed_dict=feed_dict)
                
                    # Put the batch in the FIFO ring buffer
                    update_fifo_buffer(train_x[offset:offset+residual], train_y[offset:offset+residual], episodic_images, episodic_labels,
                                        task_labels[task], args.mem_size, count_cls, episodic_filled_counter)

                elif model.imp_method == 'ER-SUBSPACE':
                    # Zero out all the grads
                    #sess.run([model.reset_er_subspace_grads, model.reset_ref_grads])
                    sess.run([model.reset_er_subspace_grads])
                    # Accumulate grads for all the tasks
                    for tt in range(task):
                        mem_offset = tt*args.mem_size*TOTAL_CLASSES
                        er_mem_indices = np.arange(mem_offset, mem_offset+args.mem_size*TOTAL_CLASSES)
                        np.random.shuffle(er_mem_indices)
                        er_train_x_batch = episodic_images[er_mem_indices]
                        er_train_y_batch = episodic_labels[er_mem_indices]
                        feed_dict = {model.x: er_train_x_batch, model.y_: er_train_y_batch,
                                     model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0,
                                     model.output_mask: logit_mask, model.task_id: task+1, model.learning_rate: args.learning_rate}
                        feed_dict[model.subspace_proj] = proj_matrices[tt]
                        sess.run(model.accum_er_subspace_grads, feed_dict=feed_dict)
                        
                    feed_dict = {model.x: train_x[offset:offset+residual], model.y_: train_y[offset:offset+residual],
                                 model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0,
                                 model.output_mask: logit_mask, model.task_id: task+1, model.learning_rate: args.learning_rate}
                    feed_dict[model.subspace_proj] = proj_matrices[task]
                    _, _, loss = sess.run([model.train_er_subspace, model.accum_er_subspace_grads, model.reg_loss], feed_dict=feed_dict)

                    # Put the batch in the FIFO ring buffer
                    update_fifo_buffer(train_x[offset:offset+residual], train_y[offset:offset+residual], episodic_images, episodic_labels,
                                       task_labels[task], args.mem_size, count_cls, episodic_filled_counter)

                if (iters % 100 == 0):
                    print('Step {:d} {:.3f}'.format(iters, loss))
                    #print('Step {:d}\t CE: {:.3f}\t Reg: {:.3f}\t TL: {:.3f}'.format(iters, entropy, reg, loss))
                    #print('Step {:d}\t Reg: {:.3f}\t TL: {:.3f}'.format(iters, reg, loss))

                if (math.isnan(loss)):
                    print('ERROR: NaNs NaNs Nans!!!')
                    sys.exit(0)

            print('\t\t\t\tTraining for Task%d done!'%(task))

            if model.imp_method == 'SUBSPACE-PROJ' and GRAD_CHECK:
                # TODO: Compute the average gradient for the task at \theta^*
                bbatch_size = 100
                grad_sum = []
                for iiters in range(train_x.shape[0]// bbatch_size):
                    offset = iiters*bbatch_size
                    feed_dict = {model.x: train_x[offset:offset+bbatch_size], model.y_: train_y[offset:offset+bbatch_size], model.keep_prob: 1.0, 
                                 model.train_phase: True, model.subspace_proj: proj_matrices[task], model.output_mask: logit_mask, model.learning_rate: args.learning_rate}
                    grad_vars, train_vars = sess.run([model.reg_gradients_vars, model.trainable_vars], feed_dict=feed_dict)
                    for v in range(len(train_vars)):
                        if iiters == 0:
                            grad_sum.append(grad_vars[v][0])
                        else:
                            grad_sum[v]  += (grad_vars[v][0] - grad_sum[v])/ iiters

                prev_task_grads.append(grad_sum)

            # Upaate the episodic memory filled counter
            if use_episodic_memory:
                episodic_filled_counter += args.mem_size * classes_per_task

            if model.imp_method == 'A-GEM' and COUNT_VIOLATIONS:
                violation_count[task] = vc
                print('Task {}: Violation Count: {}'.format(task, violation_count))
                sess.run(model.reset_violation_count, feed_dict=feed_dict)

            # Compute the inter-task updates, Fisher/ importance scores etc
            # Don't calculate the task updates for the last task
            if (task < (len(task_labels) - 1)) or MEASURE_PERF_ON_EPS_MEMORY:
                model.task_updates(sess, task, task_train_images, task_labels[task])
                print('\t\t\t\tTask updates after Task%d done!'%(task))

            if args.train_single_epoch and not args.cross_validate_mode: 
                fbatch = test_task_sequence(model, sess, datasets[0]['test'], task_labels, task, False, proj_matrices)
                ftask.append(fbatch)
                ftask = np.array(ftask)
            else:
                # List to store accuracy for all the tasks for the current trained model
                ftask = test_task_sequence(model, sess, datasets[0]['test'], task_labels, task, args.online_cross_val, proj_matrices)
                print('Task: {}, Acc: {}'.format(task, ftask))
            
            # Store the accuracies computed at task T in a list
            evals.append(ftask)

            # Reset the optimizer
            model.reset_optimizer(sess)

            #-> End for loop task

        runs.append(np.array(evals))
        # End for loop runid

    runs = np.array(runs)

    return runs

def test_task_sequence(model, sess, test_data, task_labels, task, cross_validate_mode, projection_matrices=None):
    """
    Snapshot the current performance
    """
    if TIME_MY_METHOD:
        # Only compute the training time
        return np.zeros(model.num_tasks)

    final_acc = np.zeros(model.num_tasks)
    if model.imp_method == 'PNN':
        pnn_logit_mask = np.ones([model.num_tasks, TOTAL_CLASSES])
    else:
        logit_mask = np.zeros(TOTAL_CLASSES)

    for tt, labels in enumerate(task_labels):

        # Multi-head evaluation setting
        logit_mask[:] = 0
        logit_mask[labels] = 1.0

        if not MULTI_TASK:
            if tt > task:
                return final_acc
        
        task_test_images, task_test_labels = load_task_specific_data(test_data, labels)

        if model.imp_method == 'PNN':
            pnn_train_phase = np.array(np.zeros(model.num_tasks), dtype=np.bool)
            feed_dict = {model.x: task_test_images, 
                    model.y_[tt]: task_test_labels, model.keep_prob: 1.0}
            train_phase_dict = {m_t: i_t for (m_t, i_t) in zip(model.train_phase, pnn_train_phase)}
            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, pnn_logit_mask)}
            feed_dict.update(train_phase_dict)
            feed_dict.update(logit_mask_dict)
            acc = model.accuracy[tt].eval(feed_dict = feed_dict)
        else:
            feed_dict = {model.x: task_test_images, 
                    model.y_: task_test_labels, model.keep_prob: 1.0, 
                    model.output_mask: logit_mask, model.train_phase: False}
            if model.imp_method in {'SUBSPACE-PROJ', 'ER-SUBSPACE', 'ER-PROJ-SUBSPACE', 'PROJ-ANCHOR', 'PROJ-SUBSPACE-GRADIENT-PENALTY'}:
                feed_dict[model.subspace_proj] = projection_matrices[tt]
            acc = model.accuracy.eval(feed_dict = feed_dict)

        final_acc[tt] = acc

    return final_acc


def main():
    """
    Create the model and start the training
    """

    # Get the CL arguments
    args = get_arguments()

    # Check if the network architecture is valid
    if args.arch not in VALID_ARCHS:
        raise ValueError("Network architecture %s is not supported!"%(args.arch))

    # Check if the method to compute importance is valid
    if args.imp_method not in MODELS:
        raise ValueError("Importance measure %s is undefined!"%(args.imp_method))
    
    # Check if the optimizer is valid
    if args.optim not in VALID_OPTIMS:
        raise ValueError("Optimizer %s is undefined!"%(args.optim))

    # Create log directories to store the results
    if not os.path.exists(args.log_dir):
        print('Log directory %s created!'%(args.log_dir))
        os.makedirs(args.log_dir)

    # Generate the experiment key and store the meta data in a file
    exper_meta_data = {'DATASET': 'SPLIT_MNIST',
            'NUM_RUNS': args.num_runs,
            'TRAIN_SINGLE_EPOCH': args.train_single_epoch, 
            'IMP_METHOD': args.imp_method, 
            'SYNAP_STGTH': args.synap_stgth,
            'FISHER_EMA_DECAY': args.fisher_ema_decay,
            'FISHER_UPDATE_AFTER': args.fisher_update_after,
            'OPTIM': args.optim, 
            'LR': args.learning_rate, 
            'BATCH_SIZE': args.batch_size, 
            'MEM_SIZE': args.mem_size}
    experiment_id = "SPLIT_MNIST_META_%s_%s_%r_%s-"%(args.imp_method, str(args.synap_stgth).replace('.', '_'),
            str(args.batch_size), str(args.mem_size)) + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    snapshot_experiment_meta_data(args.log_dir, experiment_id, exper_meta_data)

    # Get the subset of data depending on training or cross-validation mode
    if args.online_cross_val:
        num_tasks = K_FOR_CROSS_VAL
    else:
        num_tasks = args.num_tasks - K_FOR_CROSS_VAL

    # Load the dataset
    data_labs = [np.arange(TOTAL_CLASSES)]
    datasets = construct_split_mnist(data_labs)

    # Variables to store the accuracies and standard deviations of the experiment
    acc_mean = dict()
    acc_std = dict()

    # Reset the default graph
    tf.reset_default_graph()
    graph  = tf.Graph()
    with graph.as_default():

        # Set the random seed
        tf.set_random_seed(args.random_seed)

        # Define Input and Output of the model
        x = tf.placeholder(tf.float32, shape=[None, INPUT_FEATURE_SIZE])
        #x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        learning_rate = tf.placeholder(dtype=tf.float32, shape=())
        if args.imp_method == 'PNN':
            y_ = []
            for i in range(num_tasks):
                y_.append(tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES]))
        else:
            y_ = tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES])

        # Define the optimizer
        if args.optim == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        elif args.optim == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        elif args.optim == 'MOMENTUM':
            #base_lr = tf.constant(args.learning_rate)
            #learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - model.train_step / training_iters), OPT_POWER))
            opt = tf.train.MomentumOptimizer(learning_rate, OPT_MOMENTUM)

        # Create the Model/ contruct the graph
        model = Model(x, y_, num_tasks, opt, args.imp_method, args.synap_stgth, args.fisher_update_after, 
                args.fisher_ema_decay, learning_rate, network_arch=args.arch)

        # Set up tf session and initialize variables.
        if USE_GPU:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(
                    device_count = {'GPU': 0}
                    )

        time_start = time.time()
        with tf.Session(config=config, graph=graph) as sess:
            runs = train_task_sequence(model, sess, datasets, args)
            # Close the session
            sess.close()
        time_end = time.time()
        time_spent = time_end - time_start

    # Store all the results in one dictionary to process later
    exper_acc = dict(mean=runs)

    # If cross-validation flag is enabled, store the stuff in a text file
    if args.cross_validate_mode:
        acc_mean, acc_std = average_acc_stats_across_runs(runs, model.imp_method)
        fgt_mean, fgt_std = average_fgt_stats_across_runs(runs, model.imp_method)
        cross_validate_dump_file = args.log_dir + '/' + 'SPLIT_MNIST_%s_%s'%(args.imp_method, args.optim) + '.txt'
        with open(cross_validate_dump_file, 'a') as f:
            if MULTI_TASK:
                f.write('GPU:{} \t ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {}\n'.format(USE_GPU, args.arch, args.learning_rate, 
                    args.synap_stgth, acc_mean[-1, :].mean()))
            else:
                f.write('NUM_TASKS: {} \t EXAMPLES_PER_TASK: {} \t MEM_SIZE: {} \t ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {} (+-{})\t Fgt: {} (+-{})\t QR: {}\t Time: {}\n'.format(args.num_tasks, args.examples_per_task, args.mem_size, args.arch, args.learning_rate, 
                    args.synap_stgth, acc_mean, acc_std, fgt_mean, fgt_std, QR, str(time_spent)))

    # Store the experiment output to a file
    snapshot_experiment_eval(args.log_dir, experiment_id, exper_acc)

if __name__ == '__main__':
    main()
