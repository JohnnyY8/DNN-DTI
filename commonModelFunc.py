#coding=utf-8
import numpy as np
import tensorflow as tf

class CommonModelFunc:
  
  # Initial weight variable
  def init_weight_variable(self, varName, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    #return tf.get_variable(varName, shape, initializer = initial)
    return tf.Variable(initial_value = initial, name = varName)

  # Initial bias variable
  def init_bias_variable(self, varName, shape):
    initial = tf.constant(0.1, shape=shape)
    #return tf.get_variable(varName, shape, initializer = initial)
    return tf.Variable(initial_value = initial, name = varName)

  # Convolutional operation
  def conv2d(self, x, W, s_height, s_width, s_channels):
    # Given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape
    # x: [batch, in_height, in_width, in_channels]
    # W: [height, width, in_channels, out_channels]
    # strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
    return tf.nn.conv2d(x, W, strides=[1, s_height, s_width, s_channels], padding='SAME')

  # Max pooling operation
  def max_pool(self, x, k_height, k_width, s_height, s_width):
    # ksize: first 1 and last 1 because we don't want to take the maximum over multiple exameples or over multiple channels.
    # strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
    return tf.nn.max_pool(x, ksize=[1, k_hegith, k_width, 1], strides=[1, s_height, s_width, 1], padding='SAME')

  # Average pooling operation
  def avg_pool(self, x, k_height, k_width, s_height, s_width):
    # ksize: first 1 and last 1 because we don't want to take the maximum over multiple exameples or over multiple channels.
    # strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
    return tf.nn.avg_pool(x, ksize=[1, k_height, k_width, 1], strides=[1, s_height, s_width, 1], padding='SAME')

  # Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  def variable_summaries(self, var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

