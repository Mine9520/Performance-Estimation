#!/bin/env python
# coding:utf-8

import xdl
import xdl_runner
import numpy as np
import tensorflow as tf


def layer_norm(temp_input):
    return tf.contrib.layers.layer_norm(temp_input)


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


@xdl.tf_wrapper()
def CGC(merged_sparse_embeddings, dense_features, label, task_num=3):
    input_dense_features = tf.concat(dense_features, 1)
    input_sparse_features = tf.concat(merged_sparse_embeddings, 1)
    inputs = tf.concat([input_dense_features, input_sparse_features], 1)
    cgc_module_1_inputs = [inputs]*task_num

    cgc_module_1_task_output, cgc_module_1_shared_ouput = cgc_module(
        cgc_module_1_inputs, inputs, task_num, 256, 1)

    cgc_module_2_task_output, cgc_module_2_shared_ouput = cgc_module(
        cgc_module_1_task_output, cgc_module_1_shared_ouput, task_num, 128, 2)

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
            task_index=i, 
            tower_index=3, 
            output_dim=1, 
            activation=None) 
        for i in range(task_num)]

    # b, c, d are the parameters predicted by the neural nwtworks
    # a = tf.exp(tf.reshape(fc_tower_3_output_list[0], [-1]))*100
    b = tf.exp(tf.reshape(fc_tower_3_output_list[0], [-1])/10)*2
    c = tf.exp(tf.reshape(fc_tower_3_output_list[1], [-1]))*200
    d = tf.exp(tf.reshape(fc_tower_3_output_list[2], [-1]))*2000
    a = tf.zeros_like(b)

    price = label[:, 0]
    pv = label[:, 1]
    # A, B, C, D are the parameters predicted by least squares regression
    # A should be 0 because there should be no impressions if bid is 0
    # D should be multiplied by (the total number of impressions/the number of records in the REPLAY system)
    A = tf.zeros_like(b)
    B = label[:, 3]
    C = label[:, 4]
    D = label[:, 5]*4

    pv_nn = -d/(1+tf.pow(price/c, b)) + d
    pv_lsr = tf.where(C>0, (A-D)/(1+tf.pow(price/C, B)) + D, tf.zeros_like(C))
    weight_loss = tf.where(C>0, tf.exp(50*(2-pv_lsr/pv-pv/pv_lsr)), tf.zeros_like(C))
    weight_sample = tf.sqrt(tf.sqrt(pv))

    loss_pv = tf.losses.huber_loss(tf.log1p(pv), tf.log1p(pv_nn), weights=(1-weight_loss)*weight_sample)
    loss_b = tf.losses.huber_loss(tf.log1p(B), tf.log1p(b), weights=weight_loss*weight_sample)
    loss_c = tf.losses.huber_loss(tf.log1p(C), tf.log1p(c), weights=weight_loss*weight_sample)
    loss_d = tf.losses.huber_loss(tf.log1p(D), tf.log1p(d), weights=weight_loss*weight_sample)
    loss = loss_pv + (loss_b + loss_c + loss_d)/3

    return loss, a, b, c, d
