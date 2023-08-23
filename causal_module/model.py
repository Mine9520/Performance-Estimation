#!/bin/env python
# coding:utf-8

import xdl
import xdl_runner
import tensorflow as tf
import numpy as np
import json
import time
import datetime
import collections
import get_mean_std_tf
from xdl.python.training.tf_summary_hook import TFSummaryHookV2
from xdl.python.training.train_session import QpsMetricsHook, MetricsHook, MetricsPrinterHook
from xdl.python.utils.collections import READER_HOOKS, get_collection, TRAINABLE_VARIABLES
from xdl.python.utils.metrics import add_metrics


IS_TRAINING = True if xdl.get_config('job_type') == 'train' else False


def print_tensor_shape(var, var_name):
    var = tf.Print(var, [var_name + ' shape is: ', tf.shape(var)], first_n=1)
    return var


def print_tensor_value(var, var_name):
    var = tf.Print(var, [var_name + ' value is: ', var], first_n=10)
    return var


def layer_norm(temp_input):
    return tf.contrib.layers.layer_norm(temp_input)


def norm_dense_features(dense_features, mean_tf, std_tf):
    return (dense_features - mean_tf) / std_tf


def poisson_loss(preds, labels):
    err = preds - tf.multiply(labels, tf.math.log(preds))
    return err


def get_error_rate(preds, labels, ratio_limit=0.5):
    abs_error = tf.math.abs(tf.subtract(preds, labels))
    error_ratio = tf.divide(abs_error, labels)
    condition = tf.less(error_ratio, ratio_limit)
    in_ratio = tf.ones_like(labels, dtype=tf.float32)
    out_ratio = tf.zeros_like(labels, dtype=tf.float32)
    res_ratio = tf.where(condition, in_ratio, out_ratio)

    label_condition = tf.greater(labels, 0)
    label_greater_0 = tf.where(label_condition, tf.ones_like(labels, dtype=tf.float32),
                               tf.zeros_like(labels, dtype=tf.float32))
    mean_ratio = tf.reduce_sum(res_ratio) / tf.reduce_sum(label_greater_0)
    res_condition = tf.greater(tf.reduce_sum(label_greater_0), 0)
    return tf.where(res_condition, mean_ratio, tf.constant(0.0))


def get_abs_error(preds, labels):
    abs_error = tf.reduce_mean(tf.math.abs(preds - labels))
    return abs_error


def tf_mean_absolute_percentage_error(predict, ground_truth, weights=1):
    return tf.reduce_sum(tf.abs(predict - ground_truth)*weights) / tf.reduce_sum(tf.abs(ground_truth*weights) + 1e-6)


def make_residual_seq(seq):
    seq_offset = tf.concat([seq[:, -1:, :], seq[:, :-1, :]], 1)
    seq_sub = seq - seq_offset
    seq = tf.concat([seq, seq_sub], -1)
    return seq


def MMoE(hidden, hidden_size, n_expert, n_output):
    #define layer
    expert_layer_list = []
    for i in range(n_expert):
        expert_layer_list.append(
            tf.layers.Dense(hidden_size,activation=tf.tanh,
                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                               stddev=0.36,
                                                                               dtype=tf.float32),
                            bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                            name='expert_%d'%i)
        )
    gate_list = []
    for i in range(n_output):
        gate_list.append(
            tf.layers.Dense(n_expert,activation=tf.tanh,
                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                               stddev=0.36,
                                                                               dtype=tf.float32),
                            bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                            name='gate_layer_%d' % i)
        )
    output_layer_list = []
    for i in range(n_output):
        output_layer_list.append(
            tf.layers.Dense(1,
                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                               stddev=0.36,
                                                                               dtype=tf.float32),
                            bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                            name='out_layer_%d' % i)
        )
    #forward
    expert_hidden_list = []
    for i in range(len(expert_layer_list)):
        expert_hidden = expert_layer_list[i](hidden)
        expert_hidden_list.append(tf.expand_dims(expert_hidden, 1))

    expert_hidden_concat = tf.concat(expert_hidden_list, 1)
    out_list = []
    for i in range(n_output):
        weights = gate_list[i](hidden)
        weights = tf.expand_dims(weights, -1)
        weights = tf.nn.softmax(weights, 1)
        hidden_out = tf.reduce_sum(expert_hidden_concat*weights, 1)
        out = output_layer_list[i](hidden_out)
        out_list.append(out)
    out = tf.stack(out_list, 1)
    out = tf.squeeze(out, -1)
    # print(out.shape)
    return out


def cgc_module(task_input, shared_input, task_num, expert_dim, layer_flag):
    input_dim = task_input[0].shape.as_list()[1]
    activation = tf.nn.leaky_relu
    kernel_initializer = tf.truncated_normal_initializer(
        stddev=np.sqrt(4 / (input_dim + expert_dim)))
    bias_initializer = tf.truncated_normal_initializer(
        stddev=0.0002)

    # task experts
    task_expert_layer_list = [
        tf.layers.Dense(
            expert_dim, 
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name='task_{0}_layer_{1}_expert'.format(i, layer_flag))
        for i in range(task_num)]
    # shared experts
    shared_expert_layer = tf.layers.Dense(
        expert_dim, 
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='shared_{0}'.format(layer_flag))

    # gates for task outputs
    task_gate_layer_list = [
        tf.layers.Dense(
            2, 
            activation=tf.nn.softmax,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name='task_{0}_gate_layer_{1}'.format(i, layer_flag))
        for i in range(task_num)]

    # gates for shared output
    shared_gate_layer = tf.layers.Dense(
        task_num+1, 
        activation=tf.nn.softmax,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='shared_gate_layer_{0}'.format(layer_flag))

    # expert outputs
    task_output_list = [task_expert_layer_list[i](layer_norm(task_input[i])) 
        for i in range(task_num)]
    shared_output = shared_expert_layer(layer_norm(shared_input))

    # gate output
    task_output_weight_list = [task_gate_layer_list[i](layer_norm(task_input[i])) 
        for i in range(task_num)]
    shared_output_weight = shared_gate_layer(layer_norm(shared_input))

    # weighted task outputs
    def get_task_output(task_index):
        task_weight = tf.reshape(task_output_weight_list[task_index][:, 0], [-1, 1])
        shared_weight = tf.reshape(task_output_weight_list[task_index][:, 1], [-1, 1])
        weighted_task_output = tf.multiply(task_output_list[task_index], task_weight)
        weighted_shared_output = tf.multiply(shared_output, shared_weight)
        return tf.reduce_sum([weighted_task_output, weighted_shared_output], 0)

    cgc_task_output_list = [get_task_output(i) for i in range(task_num)]

    # weighted shared output
    shared_output_components = [
        tf.multiply(
            task_output_list[i],
            tf.reshape(shared_output_weight[:, i], [-1, 1]))
        for i in range(task_num)]

    shared_output_components.append(
        tf.multiply(
            shared_output, 
            tf.reshape(shared_output_weight[:, task_num], [-1, 1])))

    cgc_shared_output = tf.reduce_sum(shared_output_components, 0)

    return cgc_task_output_list, cgc_shared_output


@xdl.tf_wrapper(clip_grad=3.0)
def model(sparse_feature_list, dense_feature_list, label, task_num=2):
    # sparse input
    sparse_input = tf.concat(sparse_feature_list, 1)
    # dense input
    dense_input = tf.concat(dense_feature_list, 1)
    mean_tf, std_tf = get_mean_std_tf.get_mean_std_dense()
    # dense feature normalization
    dense_input_norm = norm_dense_features(dense_input, mean_tf, std_tf)
    # all inputs
    inputs = tf.concat([sparse_input, dense_input_norm], 1)
    inputs = print_tensor_value(inputs, 'inputs:')
    # ground truth
    ground_truth = label[:, 0]
    mean_tf, std_tf = get_mean_std_tf.get_mean_std_label()
    # label normalization
    ground_truth_norm = norm_dense_features(ground_truth, mean_tf, std_tf)
    print('input shape:', inputs.shape, 'ground truth norm shape:', ground_truth_norm.shape)

    cgc_module_1_inputs = [inputs]*task_num
    cgc_module_1_task_output, cgc_module_1_shared_ouput = cgc_module(cgc_module_1_inputs, inputs, task_num, 256, 1)
    cgc_module_2_task_output, cgc_module_2_shared_ouput = cgc_module(cgc_module_1_task_output, cgc_module_1_shared_ouput, task_num, 128, 2)

    output_shared_gate_layer = tf.layers.Dense(
        1, 
        activation=tf.nn.softmax,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='output_shared_gate_layer')

    def layer_forward(input_list, task_index, tower_index, output_dim, activation):
        input_dim = input_list[0].shape.as_list()[1]
        kernel_initializer = tf.truncated_normal_initializer(
            stddev=np.sqrt(4 / (output_dim+input_dim)))
        bias_initializer = tf.truncated_normal_initializer(
            stddev=0.0002)
        dense_layer = tf.layers.Dense(output_dim,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name='task_{0}_tower_{1}'.format(task_index, tower_index))
        return dense_layer(layer_norm(input_list[task_index]))


    fc_tower_1_output_list = [
        layer_forward(
            input_list=cgc_module_2_task_output, 
            task_index=i, 
            tower_index=1, 
            output_dim=64,
            activation=tf.nn.leaky_relu) 
        for i in range(task_num)]
    fc_tower_2_output_list = [
        layer_forward(
            input_list=fc_tower_1_output_list, 
            task_index=i, 
            tower_index=2, 
            output_dim=32, 
            activation=tf.nn.leaky_relu) 
        for i in range(task_num)]
    fc_tower_3_output_list = [
        layer_forward(
            input_list=fc_tower_2_output_list, 
            task_index=0, 
            tower_index=3, 
            output_dim=1, 
            activation=None),
        layer_forward(
            input_list=fc_tower_2_output_list, 
            task_index=1, 
            tower_index=3, 
            output_dim=3, 
            activation=None)]
    

    theta = tf.reshape(fc_tower_3_output_list[0], [-1])
    pre_b = -tf.exp(tf.reshape(fc_tower_3_output_list[1][:,0], [-1])/10)*2
    pre_c = tf.exp(tf.reshape(fc_tower_3_output_list[1][:,1], [-1]))*200
    pre_d = tf.exp(tf.reshape(fc_tower_3_output_list[1][:,2], [-1]))*2000
    pre_a = tf.zeros_like(pre_b)

    pv_ratio = label[:, 0] # true_label
    benchmark_bid = label[:, 1]
    real_bid = label[:, 2]

    pv_nn_real = tf.where(pre_c > 0, -pre_d / (1 + tf.pow(real_bid / pre_c, pre_b)) + pre_d, tf.zeros_like(pre_c))
    pv_nn_benchmark = tf.where(pre_c > 0, -pre_d / (1 + tf.pow(benchmark_bid / pre_c, pre_b)) + pre_d, tf.zeros_like(pre_c))
    pv_nn_ratio = tf.where(pv_nn_benchmark > 0, tf.abs(pv_nn_real) / tf.abs(pv_nn_benchmark), tf.ones_like(pv_nn_real))

    loss_1 = tf.losses.huber_loss(pv_ratio, theta)
    loss_2 = tf.losses.huber_loss(pv_ratio, pv_nn_ratio)

    loss = tf.exp(loss_1)/(tf.exp(loss_1) + tf.exp(loss_2)) * loss_1 + tf.exp(loss_2)/(tf.exp(loss_1) + tf.exp(loss_2)) * loss_2

    return loss, theta, pre_a, pre_b, pre_c, pre_d

