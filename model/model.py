# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Model defintion
"""                                        

import tensorflow as tf        
import numpy as np
import matplotlib.pyplot as plt
from utils import clone_variable_list, create_fc_layer, create_conv_layer
from utils.resnet_utils import _conv, _fc, _bn, _residual_block, _residual_block_first 
from utils.vgg_utils import vgg_conv_layer, vgg_fc_layer
from utils.normalization import OrthogNormalization

PARAM_XI_STEP = 1e-3
NEG_INF = -1e32
EPSILON = 1e-32
HYBRID_ALPHA = 0.5
TRAIN_ENTROPY_BASED_SUM = False
STIEFEL_INNP_EUC = True

def weight_variable(shape, name='fc', init_type='default'):
    """
    Define weight variables
    Args:
        shape       Shape of the bias variable tensor

    Returns:
        A tensor of size shape initialized from a random normal
    """
    with tf.variable_scope(name):
        if init_type == 'default':
            if shape[1] == 10:
                weights = tf.get_variable('classifier', shape, tf.float32, initializer=tf.initializers.orthogonal())
            else:
                weights = tf.get_variable('kernel', shape, tf.float32, initializer=tf.initializers.orthogonal())
            #weights = tf.get_variable('kernel', shape, tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            #weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='kernel')
        elif init_type == 'zero':
            weights = tf.get_variable('kernel', shape, tf.float32, initializer=tf.constant_initializer(0.1))
            #weights = tf.Variable(tf.constant(0.1, shape=shape, dtype=np.float32), name='kernel')

    return weights

def bias_variable(shape, name='fc'):
    """
    Define bias variables
    Args:
        shape       Shape of the bias variable tensor

    Returns:
        A tensor of size shape initialized from a constant
    """
    with tf.variable_scope(name):
        biases = tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.1))

    return biases
    #return tf.Variable(tf.constant(0.1, shape=shape, dtype=np.float32), name='biases') #TODO: Should we initialize it from 0

class Model:
    """
    A class defining the model
    """

    def __init__(self, x_train, y_, num_tasks, opt, imp_method, synap_stgth, fisher_update_after, fisher_ema_decay, learning_rate, 
                 network_arch='FC-S'):
        """
        Instantiate the model
        """
        # Define some placeholders which are used to feed the data to the model
        self.y_ = y_
        self.total_classes = int(self.y_.get_shape()[1])
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        if imp_method in {'A-GEM', 'ER-Ringbuffer'} and 'FC-' not in network_arch:
            self.output_mask = [tf.placeholder(dtype=tf.float32, shape=[self.total_classes]) for i in range(num_tasks)]
            self.mem_batch_size = tf.placeholder(dtype=tf.float32, shape=())
        else:
            self.output_mask = tf.placeholder(dtype=tf.float32, shape=[self.total_classes])
        self.learning_rate = learning_rate
        self.sample_weights = tf.placeholder(tf.float32, shape=[None])
        self.task_id = tf.placeholder(dtype=tf.float32, shape=())
        self.store_grad_batches = tf.placeholder(dtype=tf.float32, shape=())
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=())
        self.train_samples = tf.placeholder(dtype=tf.float32, shape=())
        self.training_iters = tf.placeholder(dtype=tf.float32, shape=())
        self.train_step = tf.placeholder(dtype=tf.float32, shape=())
        self.violation_count = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.x = x_train
        x = self.x

        # Save the arguments passed from the main script
        self.opt = opt
        self.num_tasks = num_tasks
        self.imp_method = imp_method
        self.fisher_update_after = fisher_update_after
        self.fisher_ema_decay = fisher_ema_decay
        self.network_arch = network_arch

        # A scalar variable for previous syanpse strength
        self.synap_stgth = tf.constant(synap_stgth, shape=[1], dtype=tf.float32)
        self.triplet_loss_scale = 2.1

        # Define different variables
        self.weights_old = []
        self.star_vars = []
        self.small_omega_vars = []
        self.big_omega_vars = []
        self.big_omega_riemann_vars = []
        self.fisher_diagonal_at_minima = []
        self.hebbian_score_vars = []
        self.running_fisher_vars = []
        self.tmp_fisher_vars = []
        self.max_fisher_vars = []
        self.min_fisher_vars = []
        self.max_score_vars = []
        self.min_score_vars = []
        self.normalized_score_vars = []
        self.score_vars = []
        self.normalized_fisher_at_minima_vars = []
        self.weights_delta_old_vars = []
        self.ref_grads = []
        self.accum_grads = []
        self.projected_gradients_list = []
        self.theta_not_a = [] # MER
        self.theta_i_not_w = [] # MER
        self.theta_i_a = [] # MER

        self.loss_and_train_ops_for_one_hot_vector(x, self.y_)

        # Set the operations to reset the optimier when needed
        self.reset_optimizer_ops()
    
####################################################################################
#### Internal APIs of the class. These should not be called/ exposed externally ####
####################################################################################
    def loss_and_train_ops_for_one_hot_vector(self, x, y_):
        """
        Loss and training operations for the training of one-hot vector based classification model
        """
        # Define approproate network
        if self.network_arch == 'FC-S':
            input_dim = int(x.get_shape()[1])
            layer_dims = [input_dim, 256, 256, self.total_classes]
            self.subspace_proj = tf.placeholder(dtype=tf.float32, shape=(layer_dims[-2], layer_dims[-2]))
            self.fc_variables(layer_dims)
            logits = self.fc_feedforward(x, self.weights, self.biases)

        elif self.network_arch == 'FC-B':
            input_dim = int(x.get_shape()[1])
            layer_dims = [input_dim, 2000, 2000, self.total_classes]
            self.fc_variables(layer_dims)
            logits = self.fc_feedforward(x, self.weights, self.biases)

        elif 'RESNET-' in self.network_arch:
            kernels = [7, 3, 3, 3, 3]
            filters = [64, 64, 128, 256, 512]
            strides = [1, 0, 2, 2, 2]
            self.subspace_proj = tf.placeholder(dtype=tf.float32, shape=(filters[-1], filters[-1]))
            if self.imp_method in {'A-GEM', 'ER-Ringbuffer'}:
                logits = self.resnet18_conv_feedforward(x, kernels, filters, strides)
                self.task_pruned_logits = []
                self.unweighted_entropy = []
                for i in range(self.num_tasks):
                    self.task_pruned_logits.append(tf.where(tf.tile(tf.equal(self.output_mask[i][None,:], 1.0), [tf.shape(logits)[0], 1]), logits, NEG_INF*tf.ones_like(logits)))
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=self.task_pruned_logits[i])
                    adjusted_entropy = tf.reduce_sum(tf.cast(tf.tile(tf.equal(self.output_mask[i][None,:], 1.0), [tf.shape(y_)[0], 1]), dtype=tf.float32) * y_, axis=1) * cross_entropy
                    self.unweighted_entropy.append(tf.reduce_sum(adjusted_entropy)) # We will average it later on
            else:
                logits = self.resnet18_conv_feedforward(x, kernels, filters, strides)

        if (not (self.imp_method in {'A-GEM', 'ER-Ringbuffer'}) or 'FC-' in self.network_arch):
            self.pruned_logits = tf.where(tf.tile(tf.equal(self.output_mask[None,:], 1.0), [tf.shape(logits)[0], 1]), logits, NEG_INF*tf.ones_like(logits))

        # Create list of variables for storing different measures
        # Note: This method has to be called before calculating fisher 
        # or any other importance measure
        self.init_vars()

        # Different entropy measures/ loss definitions
        if (not (self.imp_method in {'A-GEM', 'ER-Ringbuffer'}) or 'FC-' in self.network_arch):
            self.mse = 2.0*tf.nn.l2_loss(self.pruned_logits) # tf.nn.l2_loss computes sum(T**2)/ 2
            self.weighted_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_, 
                self.pruned_logits, self.sample_weights, reduction=tf.losses.Reduction.NONE))
            self.unweighted_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, 
                logits=self.pruned_logits))

        # Create operations for loss and gradient calculation
        self.loss_and_gradients(self.imp_method)

        # Store the current weights before doing a train step
        self.get_current_weights()

        # For GEM variants train ops will be defined later
        if 'GEM' not in self.imp_method:
            # Define the training operation here as Pathint ops depend on the train ops
            self.train_op()

        # Create operations to compute importance depending on the importance methods
        if self.imp_method == 'EWC':
            self.create_fisher_ops()
        elif self.imp_method == 'M-EWC':
            self.create_fisher_ops()
            self.create_pathint_ops()
            self.combined_fisher_pathint_ops()
        elif self.imp_method == 'PI':
            self.create_pathint_ops()
        elif self.imp_method == 'RWALK':
            self.create_fisher_ops()
            self.create_pathint_ops()
        elif self.imp_method == 'MAS':
            self.create_hebbian_ops()
        elif self.imp_method == 'A-GEM':
            self.create_stochastic_gem_ops()
        elif self.imp_method == 'MER':
            self.mer_beta = tf.placeholder(dtype=tf.float32, shape=())
            self.mer_gamma = tf.placeholder(dtype=tf.float32, shape=())
            self.create_mer_ops()
        elif self.imp_method == 'SUBSPACE-PROJ':
            self.create_stiefel_ops()
        elif self.imp_method == 'ER-SUBSPACE':
            self.create_er_subspace_ops()
            self.create_stiefel_ops()
        elif self.imp_method == 'PROJ-ANCHOR':
            self.online_anchor_ops()
        elif self.imp_method in ['PROJ-SUBSPACE-GP', 'ER-SUBSPACE-GP']:
            self.gradient_penalty_ops()
            self.create_er_subspace_ops()

        # Create weight save and store ops
        self.weights_store_ops()

        # Summary operations for visualization
        tf.summary.scalar("unweighted_entropy", self.unweighted_entropy)
        for v in self.trainable_vars:
            tf.summary.histogram(v.name.replace(":", "_"), v)
        self.merged_summary = tf.summary.merge_all()

        if ((self.imp_method in {'A-GEM', 'ER-Ringbuffer'}) and 'FC-' not in self.network_arch):
            self.correct_predictions = []
            self.accuracy = []
            for i in range(self.num_tasks):
                self.correct_predictions.append(tf.equal(tf.argmax(self.task_pruned_logits[i], 1), tf.argmax(y_, 1)))
                self.accuracy.append(tf.reduce_mean(tf.cast(self.correct_predictions[i], tf.float32)))
        else:
            self.correct_predictions = tf.equal(tf.argmax(self.pruned_logits, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
   
    def fc_variables(self, layer_dims):
        """
        Defines variables for a 3-layer fc network
        Args:

        Returns:
        """

        self.weights = []
        self.biases = []
        self.trainable_vars = []

        for i in range(len(layer_dims)-1):
            w = weight_variable([layer_dims[i], layer_dims[i+1]], name='fc_%d'%(i))
            b = bias_variable([layer_dims[i+1]], name='fc_%d'%(i))
            self.weights.append(w)
            self.biases.append(b)
            self.trainable_vars.append(w)
            self.trainable_vars.append(b)

    def fc_feedforward(self, h, weights, biases, apply_dropout=False):
        """
        Forward pass through a fc network
        Args:
            h               Input image (tensor)
            weights         List of weights for a fc network
            biases          List of biases for a fc network
            apply_dropout   Whether to apply droupout (True/ False)

        Returns:
            Logits of a fc network
        """
        if apply_dropout:
            h = tf.nn.dropout(h, 1) # Apply dropout on Input?
        for ii, (w, b) in enumerate(list(zip(weights, biases))[:-1]):
            if self.imp_method in {'SUBSPACE-PROJ', 'ER-SUBSPACE', 'PROJ-ANCHOR', 'PROJ-SUBSPACE-GP'} and ii == len(weights) - 2: # Apply projection at the second last layer only
                h = create_fc_layer(h, w, b, P=self.subspace_proj, OWN=False)
            else:
                h = create_fc_layer(h, w, b, OWN=False)
            if apply_dropout:
                h = tf.nn.dropout(h, 1)  # Apply dropout on hidden layers?

        # Store image features 
        self.features = h
        self.image_feature_dim = h.get_shape().as_list()[-1]
        return create_fc_layer(h, weights[-1], biases[-1], apply_relu=False)

    def fc_get_features(self, h, train_vars, apply_dropout=False):
        weights = [var for i, var in enumerate(train_vars) if i%2==0] 
        biases = [var for i, var in enumerate(train_vars) if i%2!=0]
        if apply_dropout:
            h = tf.nn.dropout(h, 1) # Apply dropout on Input?
        for ii, (w, b) in enumerate(zip(weights, biases)):
            if self.imp_method in {'PROJ-ANCHOR'} and ii == len(weights) - 1:
                h = create_fc_layer(h, w, b, P=self.subspace_proj, OWN=False)
            else:
                h = create_fc_layer(h, w, b, OWN=False)
            if apply_dropout:
                h = tf.nn.dropout(h, 1)
        return h

    def resnet18_conv_feedforward(self, h, kernels, filters, strides):
        """
        Forward pass through a ResNet-18 network

        Returns:
            Logits of a resnet-18 conv network
        """
        self.trainable_vars = []
        init_orthog =  True if (self.imp_method == 'ER-SUBSPACE') else False
        
        # Conv1
        ff = 0
        h = _conv(h, kernels[0], filters[ff], strides[ff], self.trainable_vars, orthog_init=init_orthog, name='conv_1')
        h = _bn(h, self.trainable_vars, self.train_phase, name='bn_1')

        # Conv2_x
        h = _residual_block(h, self.trainable_vars, self.train_phase, orthog_init=init_orthog, name='conv2_1')
        h = _residual_block(h, self.trainable_vars, self.train_phase, orthog_init=init_orthog, name='conv2_2')

        # Conv3_x
        ff = 2
        h = _residual_block_first(h, filters[ff], strides[ff], self.trainable_vars, self.train_phase, orthog_init=init_orthog, name='conv3_1')
        h = _residual_block(h, self.trainable_vars, self.train_phase, orthog_init=init_orthog, name='conv3_2')

        # Conv4_x
        ff = 3
        h = _residual_block_first(h, filters[ff], strides[ff], self.trainable_vars, self.train_phase, orthog_init=init_orthog, name='conv4_1')
        h = _residual_block(h, self.trainable_vars, self.train_phase, orthog_init=init_orthog, name='conv4_2')

        # Conv5_x
        ff = 4
        h = _residual_block_first(h, filters[ff], strides[ff], self.trainable_vars, self.train_phase, orthog_init=init_orthog, name='conv5_1')
        h = _residual_block(h, self.trainable_vars, self.train_phase, orthog_init=init_orthog, name='conv5_2')

        # Apply average pooling
        h = tf.reduce_mean(h, [1, 2])

        # Store the feature mappings
        self.features = h
        self.image_feature_dim = h.get_shape().as_list()[-1]

        if self.imp_method in {'SUBSPACE-PROJ', 'ER-SUBSPACE', 'PROJ-ANCHOR', 'PROJ-SUBSPACE-GP', 'ER-SUBSPACE-GP'}:
            h = tf.nn.relu(tf.matmul(h, self.subspace_proj))

        logits = _fc(h, self.total_classes, self.trainable_vars, orthog_init=init_orthog, name='classifier/fc_1')
        return logits


    def get_attribute_embedding(self, attr):
        """
        Get attribute embedding using a simple FC network

        Returns:
            Embedding vector of k x ATTR_DIMS 
        """
        w = weight_variable([self.attr_dims, self.image_feature_dim], name='attr_embed_w')
        self.trainable_vars.append(w)
        # Return the inner product of attribute matrix and weight vector. 
        return tf.matmul(attr, w) # Dimension should be TOTAL_CLASSES x image_feature_dim

    def loss_and_gradients(self, imp_method):
        """
        Defines task based and surrogate losses and their
        gradients
        Args:

        Returns:
        """
        reg = 0.0
        if imp_method in {'VAN', 'ER-Reservoir', 'ER-Ringbuffer', 'MER', 'GEM'}:
            pass
        elif imp_method in {'EWC', 'M-EWC'}:
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.normalized_fisher_at_minima_vars)])
        elif imp_method == 'PI':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.big_omega_vars)])
        elif imp_method == 'MAS':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, 
                f in zip(self.trainable_vars, self.star_vars, self.hebbian_score_vars)])
        elif imp_method == 'RWALK':
            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * (f + scr)) for w, w_star, 
                f, scr in zip(self.trainable_vars, self.star_vars, self.normalized_fisher_at_minima_vars, 
                    self.normalized_score_vars)])
        elif imp_method in {'SUBSPACE-PROJ', 'ER-SUBSPACE', 'PROJ-SUBSPACE-GP', 'ER-SUBSPACE-GP'}:
            #reg = tf.add_n([tf.norm(tf.matmul(tf.transpose(w), w) - tf.eye(tf.to_int32(w.get_shape()[-1]))) for w in self.weights])
            self.kernel_isometry = tf.add_n([tf.norm(tf.matmul(tf.transpose(tf.reshape(w, [-1, w.get_shape()[-1]])), tf.reshape(w, [-1, w.get_shape()[-1]])) - tf.eye(tf.to_int32(w.get_shape()[-1]))) for w in self.trainable_vars if 'kernel' in w.name and 'shortcut' not in w.name])
            reg = 0.0
        
        self.regularization = reg
        if imp_method != 'A-GEM': # For A-GEM we will define the losses and gradients later on
            if imp_method in {'ER-Ringbuffer'} and 'FC-' not in self.network_arch:
                self.reg_loss = tf.add_n([self.unweighted_entropy[i] for i in range(self.num_tasks)])/ self.mem_batch_size
            else:
                # Regularized training loss
                self.reg_loss = tf.squeeze(self.unweighted_entropy + self.synap_stgth * self.regularization)
                # Compute the gradients of the vanilla loss
                self.vanilla_gradients_vars = self.opt.compute_gradients(self.unweighted_entropy, 
                        var_list=self.trainable_vars)
            # Compute the gradients of regularized loss
            self.reg_gradients_vars = self.opt.compute_gradients(self.reg_loss, 
                    var_list=self.trainable_vars)

    def train_op(self):
        """
        Defines the training operation (a single step during training)
        Args:

        Returns:
        """
        if self.imp_method in {'VAN', 'ER-Reservoir', 'ER-Ringbuffer', 'SUBSPACE-PROJ', 'PROJ-ANCHOR', 'PROJ-SUBSPACE-GP'}:
            # Define training operation
            self.train = self.opt.apply_gradients(self.reg_gradients_vars)
        elif self.imp_method == 'FTR_EXT':
            # Define a training operation for the first and subsequent tasks
            self.train = self.opt.apply_gradients(self.reg_gradients_vars)
            self.train_classifier = self.opt.apply_gradients(self.reg_gradients_vars[-2:])
        else:
            # Get the value of old weights first
            with tf.control_dependencies([self.weights_old_ops_grouped]):
                # Define a training operation
                self.train = self.opt.apply_gradients(self.reg_gradients_vars)

    def init_vars(self):
        """
        Defines different variables that will be used for the
        weight consolidation
        Args:

        Returns:
        """

        for v in range(len(self.trainable_vars)):

            # List of variables for weight updates
            self.weights_old.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.weights_delta_old_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.star_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False, 
                                                  name=self.trainable_vars[v].name.rsplit(':')[0]+'_star'))

            # List of variables for pathint method
            self.small_omega_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.big_omega_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.big_omega_riemann_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

            # List of variables to store fisher information
            self.fisher_diagonal_at_minima.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

            self.normalized_fisher_at_minima_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False, dtype=tf.float32))
            self.tmp_fisher_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.running_fisher_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            # New variables for conv setting for fisher and score normalization
            self.max_fisher_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.min_fisher_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.max_score_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.min_score_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.normalized_score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            if self.imp_method == 'MAS':
                # List of variables to store hebbian information
                self.hebbian_score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            elif self.imp_method == 'MER':
                # Variables to store parameters \theta_0^A in the paper
                self.theta_not_a.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False,
                                                    name=self.trainable_vars[v].name.rsplit(':')[0]+'_theta_not_a'))
                # Variables to store parameters \theta_{i,0}^W in the paper
                self.theta_i_not_w.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False,
                                                      name=self.trainable_vars[v].name.rsplit(':')[0]+'_theta_i_not_w'))
                self.theta_i_a.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False,
                                                  name=self.trainable_vars[v].name.rsplit(':')[0]+'_theta_i_not_w'))
            elif self.imp_method in {'A-GEM', 'S-GEM', 'SUBSPACE-PROJ', 'ER-SUBSPACE', 'PROJ-ANCHOR', 'PROJ-SUBSPACE-GP', 'ER-SUBSPACE-GP'}:
                self.ref_grads.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
                self.accum_grads.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
                self.projected_gradients_list.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

    def get_current_weights(self):
        """
        Get the values of current weights
        Note: These weights are different from star_vars as those
        store the weights after training for the last task.
        Args:

        Returns:
        """
        weights_old_ops = []
        weights_delta_old_ops = []
        for v in range(len(self.trainable_vars)):
            weights_old_ops.append(tf.assign(self.weights_old[v], self.trainable_vars[v]))
            weights_delta_old_ops.append(tf.assign(self.weights_delta_old_vars[v], self.trainable_vars[v]))

        self.weights_old_ops_grouped = tf.group(*weights_old_ops)
        self.weights_delta_old_grouped = tf.group(*weights_delta_old_ops)


    def weights_store_ops(self):
        """
        Defines weight restoration operations
        Args:

        Returns:
        """
        restore_weights_ops = []
        set_star_vars_ops = []

        for v in range(len(self.trainable_vars)):
            restore_weights_ops.append(tf.assign(self.trainable_vars[v], self.star_vars[v]))

            set_star_vars_ops.append(tf.assign(self.star_vars[v], self.trainable_vars[v]))

        self.restore_weights = tf.group(*restore_weights_ops)
        self.set_star_vars = tf.group(*set_star_vars_ops)

    def reset_optimizer_ops(self):
        """
        Defines operations to reset the optimizer
        Args:

        Returns:
        """
        # Set the operation for resetting the optimizer
        self.optimizer_slots = [self.opt.get_slot(var, name) for name in self.opt.get_slot_names()\
                           for var in tf.global_variables() if self.opt.get_slot(var, name) is not None]
        self.slot_names = self.opt.get_slot_names()
        self.opt_init_op = tf.variables_initializer(self.optimizer_slots)

    def create_pathint_ops(self):
        """
        Defines operations for path integral-based importance
        Args:

        Returns:
        """
        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        update_big_omega_riemann_ops = []

        for v in range(len(self.trainable_vars)):
            # Make sure that the variables are updated before calculating delta(theta)
            with tf.control_dependencies([self.train]):
                update_small_omega_ops.append(tf.assign_add(self.small_omega_vars[v], 
                    -(self.vanilla_gradients_vars[v][0] * (self.trainable_vars[v] - self.weights_old[v]))))

            # Ops to reset the small omega
            reset_small_omega_ops.append(tf.assign(self.small_omega_vars[v], self.small_omega_vars[v]*0.0))

            if self.imp_method == 'PI':
                # Update the big omegas at the end of the task using the Eucldeian distance
                update_big_omega_ops.append(tf.assign_add(self.big_omega_vars[v], 
                    tf.nn.relu(tf.div(self.small_omega_vars[v], (PARAM_XI_STEP + tf.square(self.trainable_vars[v] - self.star_vars[v]))))))
            elif self.imp_method == 'RWALK':
                # Update the big omegas after small intervals using distance in riemannian manifold (KL-divergence)
                update_big_omega_riemann_ops.append(tf.assign_add(self.big_omega_riemann_vars[v], 
                    tf.nn.relu(tf.div(self.small_omega_vars[v], 
                        (PARAM_XI_STEP + self.running_fisher_vars[v] * tf.square(self.trainable_vars[v] - self.weights_delta_old_vars[v]))))))


        self.update_small_omega = tf.group(*update_small_omega_ops)
        self.reset_small_omega = tf.group(*reset_small_omega_ops)
        if self.imp_method == 'PI':
            self.update_big_omega = tf.group(*update_big_omega_ops)
        elif self.imp_method == 'RWALK':
            self.update_big_omega_riemann = tf.group(*update_big_omega_riemann_ops)
            self.big_omega_riemann_reset = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.big_omega_riemann_vars]

        if self.imp_method == 'RWALK':
            # For the first task, scale the scores so that division does not have an effect        
            self.scale_score = [tf.assign(s, s*2.0) for s in self.big_omega_riemann_vars]
            # To reduce the rigidity after each task the importance scores are averaged
            self.update_score = [tf.assign_add(scr, tf.div(tf.add(scr, riemm_omega), 2.0)) 
                    for scr, riemm_omega in zip(self.score_vars, self.big_omega_riemann_vars)]

            # Get the min and max in each layer of the scores
            self.get_max_score_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_max(scr, keepdims=True)), 
                axis=0)) for var, scr in zip(self.max_score_vars, self.score_vars)]
            self.get_min_score_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_min(scr, keepdims=True)), 
                axis=0)) for var, scr in zip(self.min_score_vars, self.score_vars)]
            self.max_score = tf.reduce_max(tf.convert_to_tensor(self.max_score_vars))
            self.min_score = tf.reduce_min(tf.convert_to_tensor(self.min_score_vars))
            with tf.control_dependencies([self.max_score, self.min_score]):
                self.normalize_scores = [tf.assign(tgt, (var - self.min_score)/ (self.max_score - self.min_score + EPSILON)) 
                        for tgt, var in zip(self.normalized_score_vars, self.score_vars)]

            # Sparsify all the layers except last layer
            sparsify_score_ops = []
            for v in range(len(self.normalized_score_vars) - 2):
                sparsify_score_ops.append(tf.assign(self.normalized_score_vars[v], 
                    tf.nn.dropout(self.normalized_score_vars[v], self.keep_prob)))

            self.sparsify_scores = tf.group(*sparsify_score_ops)

    def create_fisher_ops(self):
        """
        Defines the operations to compute online update of Fisher
        Args:

        Returns:
        """
        ders = tf.gradients(self.unweighted_entropy, self.trainable_vars)
        fisher_ema_at_step_ops = []
        fisher_accumulate_at_step_ops = []

        # ops for running fisher
        self.set_tmp_fisher = [tf.assign_add(f, tf.square(d)) for f, d in zip(self.tmp_fisher_vars, ders)]

        # Initialize the running fisher to non-zero value
        self.set_initial_running_fisher = [tf.assign(r_f, s_f) for r_f, s_f in zip(self.running_fisher_vars,
                                                                           self.tmp_fisher_vars)]

        self.set_running_fisher = [tf.assign(f, (1 - self.fisher_ema_decay) * f + (1.0/ self.fisher_update_after) *
                                    self.fisher_ema_decay * tmp) for f, tmp in zip(self.running_fisher_vars, self.tmp_fisher_vars)]

        self.get_fisher_at_minima = [tf.assign(var, f) for var, f in zip(self.fisher_diagonal_at_minima,
                                                                         self.running_fisher_vars)]

        self.reset_tmp_fisher = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.tmp_fisher_vars]

        # Get the min and max in each layer of the Fisher
        self.get_max_fisher_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_max(scr, keepdims=True)), axis=0)) 
                for var, scr in zip(self.max_fisher_vars, self.fisher_diagonal_at_minima)]
        self.get_min_fisher_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_min(scr, keepdims=True)), axis=0)) 
                for var, scr in zip(self.min_fisher_vars, self.fisher_diagonal_at_minima)]
        self.max_fisher = tf.reduce_max(tf.convert_to_tensor(self.max_fisher_vars))
        self.min_fisher = tf.reduce_min(tf.convert_to_tensor(self.min_fisher_vars))
        with tf.control_dependencies([self.max_fisher, self.min_fisher]):
            self.normalize_fisher_at_minima = [tf.assign(tgt, 
                (var - self.min_fisher)/ (self.max_fisher - self.min_fisher + EPSILON)) 
                    for tgt, var in zip(self.normalized_fisher_at_minima_vars, self.fisher_diagonal_at_minima)]

        self.clear_attr_embed_reg = tf.assign(self.normalized_fisher_at_minima_vars[-2], tf.zeros_like(self.normalized_fisher_at_minima_vars[-2]))

        # Sparsify all the layers except last layer
        sparsify_fisher_ops = []
        for v in range(len(self.normalized_fisher_at_minima_vars) - 2):
            sparsify_fisher_ops.append(tf.assign(self.normalized_fisher_at_minima_vars[v],
                tf.nn.dropout(self.normalized_fisher_at_minima_vars[v], self.keep_prob)))

        self.sparsify_fisher = tf.group(*sparsify_fisher_ops)

    def combined_fisher_pathint_ops(self):
        """
        Define the operations to refine Fisher information based on parameters convergence
        Args:

        Returns:
        """
        #self.refine_fisher_at_minima = [tf.assign(f, f*(1.0/(s+1e-12))) for f, s in zip(self.fisher_diagonal_at_minima, self.small_omega_vars)]
        self.refine_fisher_at_minima = [tf.assign(f, f*tf.exp(-100.0*s)) for f, s in zip(self.fisher_diagonal_at_minima, self.small_omega_vars)]


    def create_hebbian_ops(self):
        """
        Define operations for hebbian measure of importance (MAS)
        """
        # Compute the gradients of mse loss
        self.mse_gradients = tf.gradients(self.mse, self.trainable_vars)
        #with tf.control_dependencies([self.mse_gradients]):
        # Keep on adding gradients to the omega
        self.accumulate_hebbian_scores = [tf.assign_add(omega, tf.abs(grad)) for omega, grad in zip(self.hebbian_score_vars, self.mse_gradients)]
        # Average across the total images
        self.average_hebbian_scores = [tf.assign(omega, omega*(1.0/self.train_samples)) for omega in self.hebbian_score_vars]
        # Reset the hebbian importance variables
        self.reset_hebbian_scores = [tf.assign(omega, tf.zeros_like(omega)) for omega in self.hebbian_score_vars]


    def create_er_subspace_ops(self):
        """
        """
        # Accumulate and normalize the gradients
        self.kernel_grads = [grad for grad, var in self.reg_gradients_vars if 'kernel' in var.name and 'shortcut' not in var.name]
        self.accum_er_subspace_grads = [self.accum_grads[i].assign_add(grad/ self.task_id) for i, (grad, var) in enumerate(self.reg_gradients_vars)]
        self.reset_er_subspace_grads = [v.assign(tf.zeros_like(v)) for v in self.accum_grads]
        with tf.control_dependencies(self.accum_er_subspace_grads):
            self.train_er_subspace = self.opt.apply_gradients(zip(self.accum_grads, self.trainable_vars)) 


    def create_stiefel_ops(self):
        """
        Create stiefel ops
        Step 1: Project gradient onto the tangent space at current param
        Step 2: Use Cayley transform to project the vector in tangent space onto the manifold
        Assumption: Each matrix has dims [In x Out], where In => Out.
        """
        # Projection onto tangent space => U = A.W, 
        # Iterative Cayley transform
        # Y(a) = W + 0.5*a*A.(W + Y(a)), a is a hyperparameter
        # where 
        # If canonical inner prod, A = G.W^T - W.G^T 
        # If Euclidean inner prod, A = B - B^T, B = GW^T - 0.5*W.(W^T.G.W^T)
        def matrix_one_norm(a):
            return tf.reduce_max(tf.norm(a, ord=1, axis=1))

        update_stiefel_vars = []
        if self.imp_method == 'ER-SUBSPACE':
            grad_vars = zip(self.accum_grads, self.trainable_vars)
        elif self.imp_method == 'SUBSPACE-PROJ':
            grad_vars = self.reg_gradients_vars
        
        for i, (grad, var) in enumerate(grad_vars):
            if 'kernel' in var.name:
                shape = var.get_shape().as_list()
                W = tf.reshape(var, [-1, shape[-1]])
                G = tf.reshape(grad, [-1, shape[-1]])
                if STIEFEL_INNP_EUC:
                    gxT = tf.matmul(G, W, transpose_b=True)
                    xTgxT = tf.matmul(W, gxT, transpose_a=True)
                    A_hat = gxT - 0.5*tf.matmul(W, xTgxT)
                    A = A_hat - tf.transpose(A_hat)
                else: # Canonical inner prod
                    A = tf.matmul(G, W, transpose_b=True) - tf.matmul(W, G, transpose_b=True)
                U = tf.matmul(A, W) # [In x Out] # Tangent vector
                t = 0.5 * 2 / (matrix_one_norm(A) + 1e-8)
                alpha = tf.minimum(self.learning_rate, t)
                Y_0 = W - alpha*U
                Y_1 = W - alpha*tf.matmul(A, 0.5*(W+Y_0))
                Y_2 = W - alpha*tf.matmul(A, 0.5*(W+Y_1))
                update_stiefel_vars.append(var.assign(tf.reshape(Y_2, shape)))
            else:
                apply_grads = self.opt.apply_gradients(zip([grad], [var]))
                update_stiefel_vars.append(apply_grads)

        self.train_stiefel = tf.group(*update_stiefel_vars)


    def create_svb_ops(self, epsilon=5e-2):
        """
        Bound the singular values of weight matrices
        """
        svb_ops = []
        for w in self.trainable_vars:
            if 'kernel' in w.name:
                shape = w.get_shape().as_list()
                w_reshaped = tf.reshape(w, [-1, shape[-1]])
                D, U, V = tf.linalg.svd(w_reshaped)
                D_clipped = tf.clip_by_value(D, 1/(1+epsilon), 1+epsilon)
                svb_ops.append(tf.assign(w, tf.reshape(tf.matmul(U, tf.matmul(tf.linalg.diag(D_clipped), V, adjoint_b=True)), shape)))
        self.update_weights_svb = tf.group(*svb_ops)


    def subspace_proj_ops(self):
        def proj_u(u, v):
            u_hat = u/ tf.reduce_sum(tf.square(u))
            scal = tf.reduce_sum(tf.multiply(u, v))
            return scal*u_hat

        # Compute and store gradients in the orthogonal compliment space
        p_perp_grads = [grad for grad, var in self.reg_gradients_vars]
        self.store_ref_grads = [tf.assign(ref, grad) for ref, grad in zip(self.ref_grads, p_perp_grads)]

        store_proj_grad_ops = []
        # Compute the final gradients
        for i, (t_grad, var) in enumerate(self.reg_gradients_vars):
            if 'classifier' in var.name:
                store_proj_grad_ops.append(tf.assign(self.projected_gradients_list[i], t_grad))
            else:
                ref_grad = self.ref_grads[i]
                grad = t_grad - proj_u(ref_grad, t_grad)
                store_proj_grad_ops.append(tf.assign(self.projected_gradients_list[i], grad))
        
        self.store_proj_grads = tf.group(*store_proj_grad_ops)
        with tf.control_dependencies([self.store_proj_grads]):
            self.train_subspace_proj = self.opt.apply_gradients(zip(self.projected_gradients_list, self.trainable_vars))

    def online_anchor_ops(self):
        # Temporary gradient update in the orthogonal compliment space
        tmp_params = [var - self.learning_rate*grad for grad, var in self.reg_gradients_vars]

        tmp_features = self.fc_get_features(self.x, tmp_params[:-2])
        original_features = self.fc_get_features(self.x, self.trainable_vars[:-2]) # Not going to use the classifier weights
        
        self.anchor_loss = tf.reduce_mean(tf.reduce_sum((tmp_features-original_features)**2, axis=1))
        self.anchor_grad_vars = self.opt.compute_gradients(self.anchor_loss, var_list=self.trainable_vars[:-2])
        p_perp_grads = [grad for grad, var in self.anchor_grad_vars]
        self.store_ref_grads = [tf.assign(ref, grad) for ref, grad in zip(self.ref_grads[:-2], p_perp_grads)]

        # Compute the final gradients
        store_proj_grad_ops = []
        for i, (t_grad, var) in enumerate(self.reg_gradients_vars):
            if 'classifier' in var.name:
                store_proj_grad_ops.append(tf.assign(self.projected_gradients_list[i], t_grad))
            else:
                grad = t_grad + self.synap_stgth*self.ref_grads[i]
                store_proj_grad_ops.append(tf.assign(self.projected_gradients_list[i], grad))
        store_proj_grads = tf.group(*store_proj_grad_ops)
        with tf.control_dependencies([store_proj_grads]):
            self.train_subspace_proj = self.opt.apply_gradients(zip(self.projected_gradients_list, self.trainable_vars))

    def gradient_penalty_ops(self):
        
        self.reset_er_subspace_gp_grads = [v.assign(tf.zeros_like(v)) for v in self.accum_grads]
        # Compute and store gradients in the orthogonal compliment space
        p_perp_grads = [grad for grad, var in self.reg_gradients_vars]
        self.store_ref_grads = [tf.assign(ref, grad) for ref, grad in zip(self.ref_grads, p_perp_grads)]

        # Compute the gradient penalty
        gradient_p = []
        for i, (grad, var) in enumerate(self.reg_gradients_vars):
            grad_flat = tf.reshape(grad, [-1])
            ref_grad_flat = tf.reshape(self.ref_grads[i], [-1])
            gradient_p.append(tf.reduce_sum(tf.multiply(grad_flat, ref_grad_flat)))

        self.gradient_penalty = tf.add_n([gp for gp in gradient_p[:-2]])
        with tf.control_dependencies([self.gradient_penalty]):
            self.gp_total_loss = tf.squeeze(self.reg_loss + self.synap_stgth*self.gradient_penalty)
            self.gp_gradients_vars = self.opt.compute_gradients(self.gp_total_loss, var_list=self.trainable_vars)
            self.train_gp = self.opt.apply_gradients(self.gp_gradients_vars)
            self.accum_er_subspace_gp_grads = [self.accum_grads[i].assign_add(grad/ self.task_id) for i, (grad, var) in enumerate(self.gp_gradients_vars)]
            with tf.control_dependencies(self.accum_er_subspace_gp_grads):
                self.train_er_gp = self.opt.apply_gradients(zip(self.accum_grads, self.trainable_vars)) 

    def create_mer_ops(self):
        """
        Define operations for Meta-Experience replay
        """
        # Operation to store \theta_0^A
        self.store_theta_not_a = [tf.assign(var, val) for var, val in zip(self.theta_not_a, self.trainable_vars)]
        # Operation to store \theta_{i,0}^W
        self.store_theta_i_not_w = [tf.assign(var, val) for var, val in zip(self.theta_i_not_w, self.trainable_vars)]
        # Operation to store \theta_i^W
        self.store_theta_i_a = [tf.assign(var, val) for var, val in zip(self.theta_i_a, self.trainable_vars)]
        # With in batch reptile update
        self.with_in_batch_reptile_update = [tf.assign(var, val + self.mer_beta * (var - val)) for var, val in zip(self.trainable_vars, self.theta_i_not_w)]
        # Across the batch reptile update
        self.across_batch_reptile_update = [tf.assign(var, val1 + self.mer_gamma * (val2 - val1)) for var, val1, val2 in zip(self.trainable_vars, self.theta_not_a, self.theta_i_a)]

    def create_stochastic_gem_ops(self):
        """
        Define operations for Stochastic GEM
        """
        if 'FC-' in self.network_arch or self.imp_method == 'S-GEM':
            self.agem_loss = self.unweighted_entropy
        else:
            self.agem_loss = tf.add_n([self.unweighted_entropy[i] for i in range(self.num_tasks)])/ self.mem_batch_size

        ref_grads = tf.gradients(self.agem_loss, self.trainable_vars)
        # Reference gradient for previous tasks
        self.store_ref_grads = [tf.assign(ref, grad) for ref, grad in zip(self.ref_grads, ref_grads)]
        flat_ref_grads =  tf.concat([tf.reshape(grad, [-1]) for grad in self.ref_grads], 0)
        # Grandient on the current task
        task_grads = tf.gradients(self.agem_loss, self.trainable_vars)
        flat_task_grads = tf.concat([tf.reshape(grad, [-1]) for grad in task_grads], 0)
        with tf.control_dependencies([flat_task_grads]):
            dotp = tf.reduce_sum(tf.multiply(flat_task_grads, flat_ref_grads))
            ref_mag = tf.reduce_sum(tf.multiply(flat_ref_grads, flat_ref_grads))
            proj = flat_task_grads - ((dotp/ ref_mag) * flat_ref_grads)
            self.reset_violation_count = self.violation_count.assign(0)
            def increment_violation_count():
                with tf.control_dependencies([tf.assign_add(self.violation_count, 1)]):
                    return tf.identity(self.violation_count)
            self.violation_count = tf.cond(tf.greater_equal(dotp, 0), lambda: tf.identity(self.violation_count), increment_violation_count)
            projected_gradients = tf.cond(tf.greater_equal(dotp, 0), lambda: tf.identity(flat_task_grads), lambda: tf.identity(proj))
            # Convert the flat projected gradient vector into a list
            offset = 0
            store_proj_grad_ops = []
            for v in self.projected_gradients_list:
                shape = v.get_shape()
                v_params = 1
                for dim in shape:
                    v_params *= dim.value
                store_proj_grad_ops.append(tf.assign(v, tf.reshape(projected_gradients[offset:offset+v_params], shape)))
                offset += v_params
            self.store_proj_grads = tf.group(*store_proj_grad_ops)
            # Define training operations for the tasks > 1
            with tf.control_dependencies([self.store_proj_grads]):
                self.train_subseq_tasks = self.opt.apply_gradients(zip(self.projected_gradients_list, self.trainable_vars))

        # Define training operations for the first task
        self.first_task_gradients_vars = self.opt.compute_gradients(self.agem_loss, var_list=self.trainable_vars)
        self.train_first_task = self.opt.apply_gradients(self.first_task_gradients_vars)


#################################################################################
#### External APIs of the class. These will be called/ exposed externally #######
#################################################################################
    def reset_optimizer(self, sess):
        """
        Resets the optimizer state
        Args:
            sess        TF session

        Returns:
        """

        # Call the reset optimizer op
        sess.run(self.opt_init_op)

    def set_active_outputs(self, sess, labels):
        """
        Set the mask for the labels seen so far
        Args:
            sess        TF session
            labels      Mask labels

        Returns:
        """
        new_mask = np.zeros(self.total_classes)
        new_mask[labels] = 1.0
        """
        for l in labels:
            new_mask[l] = 1.0
        """
        sess.run(self.output_mask.assign(new_mask))

    def init_updates(self, sess):
        """
        Initialization updates
        Args:
            sess        TF session

        Returns:
        """
        # Set the star values to the initial weights, so that we can calculate
        # big_omegas reliably
        sess.run(self.set_star_vars)

    def task_updates(self, sess, task, train_x, train_labels, num_classes_per_task=10, online_cross_val=False):
        """
        Updates different variables when a task is completed
        Args:
            sess                TF session
            task                Task ID
            train_x             Training images for the task 
            train_labels        Labels in the task
        Returns:
        """
        if self.imp_method in {'VAN', 'ER-Reservoir', 'ER-Ringbuffer', 'SUBSPACE-PROJ', 'ER-SUBSPACE', 'PROJ-ANCHOR', 'PROJ-SUBSPACE-GP'}:
            # We'll store the current parameters at the end of this function
            pass
        elif self.imp_method == 'EWC':
            # Get the fisher at the end of a task
            sess.run(self.get_fisher_at_minima)
            # Normalize the fisher
            sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
            sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
            # Don't regularize over the attribute-embedding vectors
            #sess.run(self.clear_attr_embed_reg)
            # Reset the tmp fisher vars
            sess.run(self.reset_tmp_fisher)
        elif self.imp_method == 'M-EWC':
            # Get the fisher at the end of a task
            sess.run(self.get_fisher_at_minima)
            # Refine Fisher based on the convergence info
            sess.run(self.refine_fisher_at_minima)
            # Normalize the fisher
            sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
            sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
            # Reset the tmp fisher vars
            sess.run(self.reset_tmp_fisher)
            # Reset the small_omega_vars
            sess.run(self.reset_small_omega)
        elif self.imp_method == 'PI':
            # Update big omega variables
            sess.run(self.update_big_omega)
            # Reset the small_omega_vars because big_omega_vars are updated before it
            sess.run(self.reset_small_omega)
        elif self.imp_method == 'RWALK':
            if task == 0:
                # If first task then scale by a factor of 2, so that subsequent averaging does not hurt
                sess.run(self.scale_score)
            # Get the updated importance score
            sess.run(self.update_score)
            # Normalize the scores 
            sess.run([self.get_max_score_vars, self.get_min_score_vars])
            sess.run([self.min_score, self.max_score, self.normalize_scores])
            # Sparsify scores
            """
            # TODO: Tmp remove this?
            kp = 0.8 + (task*0.5)
            if (kp > 1):
                kp = 1.0
            """
            #sess.run(self.sparsify_scores, feed_dict={self.keep_prob: kp})
            # Get the fisher at the end of a task
            sess.run(self.get_fisher_at_minima)
            # Normalize fisher
            sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
            sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
            # Sparsify fisher
            #sess.run(self.sparsify_fisher, feed_dict={self.keep_prob: kp})
            # Store the weights
            sess.run(self.weights_delta_old_grouped)
            # Reset the small_omega_vars because big_omega_vars are updated before it
            sess.run(self.reset_small_omega)
            # Reset the big_omega_riemann because importance score is stored in the scores array
            sess.run(self.big_omega_riemann_reset)
            # Reset the tmp fisher vars
            sess.run(self.reset_tmp_fisher)
        elif self.imp_method == 'MAS':
            # zero out any previous values
            sess.run(self.reset_hebbian_scores)
            # Logits mask
            logit_mask = np.zeros(self.total_classes)
            logit_mask[train_labels] = 1.0

            # Loop over the entire training dataset to compute the parameter importance
            batch_size = 10
            num_samples = train_x.shape[0]
            for iters in range(num_samples// batch_size):
                offset = iters * batch_size
                sess.run(self.accumulate_hebbian_scores, feed_dict={self.x: train_x[offset:offset+batch_size], self.keep_prob: 1.0, 
                    self.output_mask: logit_mask, self.train_phase: False})

            # Average the hebbian scores across the training examples
            sess.run(self.average_hebbian_scores, feed_dict={self.train_samples: num_samples})
            
        # Store current weights
        self.init_updates(sess)

    def restore(self, sess):
        """
        Restore the weights from the star variables
        Args:
            sess        TF session

        Returns:
        """
        sess.run(self.restore_weights)
