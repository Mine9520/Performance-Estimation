#!/bin/env python
# coding:utf-8

import xdl
import tensorflow as tf
import numpy as np
import get_mean_std_tf

IS_TRAINING = True if xdl.get_config('job_type') == 'train' else False


def print_tensor_value(var, var_name):
    var = tf.Print(var, [var_name + ' value is: ', var], first_n=10)
    return var


def norm_dense_features(dense_features, mean_tf, std_tf):
    # mean_tf = tf.reduce_mean(dense_features, 0)
    # std_tf = tf.math.reduce_std(dense_features, 0)

    # normed_dense_features = tf.divide(tf.subtract(dense_features, mean_tf), std_tf + 0.00001)
    # normed_dense_features = tf.where(tf.is_nan(normed_dense_features),
    #                                  tf.ones_like(normed_dense_features) * 0.0,
    #                                  normed_dense_features)
    # normed_dense_features = tf.clip_by_value(normed_dense_features, -3, 3)
    return (dense_features - mean_tf) / std_tf


def tf_mean_absolute_percentage_error(predict, ground_truth, weights=1):
    return tf.reduce_sum(tf.abs(predict - ground_truth) * weights) / tf.reduce_sum(tf.abs(ground_truth * weights) + 1e-6)


def make_residual_seq(seq):
    seq_offset = tf.concat([seq[:, -1:, :], seq[:, :-1, :]], 1)
    seq_sub = seq - seq_offset
    seq = tf.concat([seq, seq_sub], -1)
    return seq


@xdl.tf_wrapper(clip_grad=3.0, is_training=IS_TRAINING)
# id feature(X), time id, impact factor(W), input:impression sequence(Y), 
# label:value of today's(Y), weight on term or sample(not sure)
def model(sparse_feature_list, sparse_time_feature_list, dense_feature_list, 
          dense_seq_feature_list, ground_truth, sample_weight):
    # print and get
    # get label
    ground_truth = print_tensor_value(ground_truth, 'ground_truth')
    sample_weight = sample_weight[:, 0]
    # get number of imp: for evaluate
    imp = ground_truth[:, 0]
    imp = print_tensor_value(imp, 'imp')
    print('check here', imp.shape, ground_truth.shape)
    # sparse input: id feature + time id
    sparse_input = tf.concat(sparse_feature_list + sparse_time_feature_list, 1)
    sparse_input = print_tensor_value(sparse_input, 'id_embed_input')
    # sparse time input: time_id
    sparse_time_input = tf.concat(sparse_time_feature_list, 1)
    sparse_time_input = print_tensor_value(sparse_time_input, 'time_embed_input')
    # dense input: impact factor
    dense_input = tf.concat(dense_feature_list, 1)
    dense_input = print_tensor_value(dense_input, 'dense_input')
    # get mean,std of dense features
    mean_tf, std_tf = get_mean_std_tf.get_mean_std_dense()
    mean_tf = print_tensor_value(mean_tf, 'mean_dense')
    std_tf = print_tensor_value(std_tf, 'std_dense')
    # standardise dense features
    dense_input_norm = norm_dense_features(dense_input, mean_tf, std_tf)
    dense_input_norm = print_tensor_value(dense_input_norm, 'dense_input_norm')
    # input: impression sequence
    # batch * seq * 1
    for i in range(len(dense_seq_feature_list)):
        dense_seq_feature_list[i] = tf.expand_dims(dense_seq_feature_list[i], -1)
    # input
    dense_seq_input = tf.concat(dense_seq_feature_list, -1)
    dense_seq_input = print_tensor_value(dense_seq_input, 'dense_input_tensor')
    # get mean,std of impression sequence
    mean_tf, std_tf = get_mean_std_tf.get_mean_std_dense_seq(1)
    mean_tf = print_tensor_value(mean_tf, 'mean_seq_feature')
    std_tf = print_tensor_value(std_tf, 'std_seq_feature')
    # check the dimension of inputs
    print(mean_tf.shape, std_tf.shape, dense_seq_input.shape, len(dense_seq_feature_list))
    # standardise inputs
    dense_seq_input_norm = norm_dense_features(dense_seq_input, mean_tf, std_tf)
    dense_seq_input_norm = print_tensor_value(dense_seq_input_norm, 'dense_tensor_norm')
    # get residual of inputs
    # dense_seq_input_norm = make_residual_seq(dense_seq_input_norm)
    # all inputs
    # features
    fix_feature = tf.concat([sparse_input, dense_input_norm], 1)
    # ground truth
    # get mean,std of label
    mean_label_tf, std_label_tf = get_mean_std_tf.get_mean_std_label()
    mean_label_tf = print_tensor_value(mean_label_tf, 'mean_ground_truth')
    std_label_tf = print_tensor_value(std_label_tf, 'std_ground_truth')
    # standardise label
    ground_truth_norm = norm_dense_features(ground_truth, mean_label_tf, std_label_tf)
    ground_truth_norm = print_tensor_value(ground_truth_norm, 'ground_truth_tensor_norm')
    print(dense_seq_input_norm.shape, fix_feature.shape, ground_truth_norm.shape)
    # network structure
    hidden_size = 256
    # 2 layers nn
    fc_layer_2 = tf.layers.Dense(hidden_size, activation=tf.tanh,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                    stddev=0.36,
                                                                                    dtype=tf.float32),
                                 bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                 name='model_1_dense2')
    fc_layer_3 = tf.layers.Dense(hidden_size, activation=tf.tanh,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                    stddev=0.36,
                                                                                    dtype=tf.float32),
                                 bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                 name='model_1_dense3')
    out_layer_labels = tf.layers.Dense(1,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                          stddev=0.36,
                                                                                          dtype=tf.float32),
                                       bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                       name='model_1_output')
    # lSTM cell
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, 
                                       initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                   stddev=0.36,
                                                                                   dtype=tf.float32))
    # attention
    attention_layer = tf.layers.Dense(1,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                         stddev=0.36,
                                                                                         dtype=tf.float32),
                                      bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                      name='model_attention')

    # LSTM: output, cell, hidden
    x, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,  # cell you have chosen
        dense_seq_input_norm,  # input
        initial_state=None,  # the initial hidden state
        dtype=tf.float32,  # must given if set initial_state = None
        time_major=False  # False: (batch, time step, input); True: (time step, batch, input)
    )
    # feed the output of LSTM into attention
    attention_weight = attention_layer(x)
    attention_weight = tf.nn.softmax(attention_weight, 1)
    # attention pooling
    x = tf.reduce_sum(x * attention_weight, 1)
    # concat other features and feed nn
    x = tf.concat([x, fix_feature], -1)
    x = fc_layer_2(x)
    x = fc_layer_3(x)
    # prediction = out_layer_labels(tf.concat([x, fix_feature], 1))
    prediction = out_layer_labels(tf.concat([x, sparse_time_input], 1))
    # prediction = out_layer_labels(x)
    prediction = print_tensor_value(prediction, 'prediction nn output')
    # prediction de-standardize
    prediction_re_norm = prediction * std_label_tf + mean_label_tf
    prediction_re_norm = print_tensor_value(prediction_re_norm, 'prediction_re_norm')
    # evaluate and loss
    pre_imp = prediction_re_norm[:, 0]
    # wmape with sample weight
    wmape_imp = tf_mean_absolute_percentage_error(pre_imp, imp, weights=sample_weight if sample_weight is not None else 1)
    sum_wmape_threshold = 1
    sum_pre_wmape = wmape_imp
    loss_log_mape = tf.math.log(wmape_imp + 1)
    # mse with sample weight
    loss_mse = tf.losses.mean_squared_error(pre_imp, imp, weights=sample_weight if sample_weight is not None else 1)
    # loss selection
    loss = tf.cond(tf.less(sum_pre_wmape, sum_wmape_threshold), lambda:loss_log_mape, lambda:loss_mse)
    loss += loss_mse
    prediction_final = tf.concat([tf.reshape(pre_imp, [-1, 1])], 1)
    return loss, loss_mse, prediction_final, wmape_imp, sum_pre_wmape, tf.reduce_sum(imp), tf.reduce_sum(pre_imp)
