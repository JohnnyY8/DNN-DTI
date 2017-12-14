#coding=utf-8
import os
import math
import numpy as np
import tensorflow as tf

from commonModelFunc import *

class BaseCNNModel(CommonModelFunc):

  def __init__(self, FLAGS, insDataPro):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro

  # Building CNN graph for base model
  def buildBaseCNNModelGraph(self):
    with tf.device("/gpu:0"):
      self.keepProb = tf.placeholder(tf.float32, name = "keepProb")
      self.init = tf.global_variables_initializer()
  
      self.xData = tf.placeholder(tf.float32, [self.FLAGS.batchSize, 1956, self.FLAGS.maxInputChannels], name = "xData")
      self.xInput = tf.reshape(self.xData, [-1, self.FLAGS.batchSize, 1956, self.FLAGS.maxInputChannels])
      self.yLabel = tf.placeholder(tf.float32, [1, 2], name = "yLabel")

      with tf.variable_scope("conv1Layer"):
        conv1KHeight = 1
        conv1KWidth = self.FLAGS.conv1KWidth
        conv1SHeight = 1
        conv1SWidth = self.FLAGS.conv1SWidth
        num4InputChannels = self.FLAGS.maxInputChannels
        num4OutputChannels = self.FLAGS.num4OutputChannels
        wConv1 = self.init_weight_variable("wConv1", [conv1KHeight, conv1KWidth, num4InputChannels, num4OutputChannels])
        bConv1 = self.init_bias_variable("bConv1", [num4OutputChannels])
        hConv1 = tf.nn.relu(self.conv2d(self.xInput, wConv1, conv1SHeight, conv1SWidth, num4InputChannels) + bConv1)

      with tf.variable_scope("roiPoolingLayer"):
        shape4hConv1 = hConv1.get_shape().as_list()
        num4FC = self.FLAGS.num4FC
        num4EachFM = num4FC / num4OutputChannels
        pool1KHeight = 1
        print "num4EachFM:", num4EachFM
        pool1KWidth = math.ceil(shape4hConv1[1] / num4EachFM)
        pool1SHeight = 1
        pool1SWidth = math.ceil(shape4hConv1[1] / num4EachFM)
        hROIPooling = self.avg_pool(hConv1, pool1KHeight, pool1KWidth, pool1SHeight, pool1SWidth)  # 平均池化 拼接

      ## 这里考虑一下是否还需要加fc
      #with tf.variable_scope("softmaxLayer"):
      #  wOutput = init_weight_variable([self.num4fc, 2])
      #  bOutput = init_bias_variable([2])
      #  hOutput = tf.matmul(hROIPooling, wOutput) + bOutput
      #  yOutput = tf.nn.softmax(hOutput)

      #with tf.variable_scope("costLayer"):
      #  predPro4PandN = tf.reshape(tf.reduce_sum(yOutput, reduction_indices = [0]), [-1, 2])
      #  predPro4P = tf.matmul(predPro4PandN, tf.constant([[0.], [1.]]))
      #  predPro4N = tf.matmul(predPro4PandN, tf.constant([[1.], [0.]]))
      #  predPro4PandNwithLabel = tf.reshape(tf.reduce_sum(self.yLabel * yOutput, reduction_indices = [0]), [-1, 2])
      #  predPro4PwithLabel = tf.matmul(predPro4PandNwithLabel, tf.constant([[0.], [1.]]))
      #  predPro4NwithLabel = tf.matmul(predPro4PandNwithLabel, tf.constant([[1.], [0.]]))
      #  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hOutput, labels = self.yLabel)) - n_weight * predPro4NwithLabel
      #  trainStep = tf.train.AdamOptimizer(learningRate).minimize(cost)  # GradientDescentOptimizer AdadeltaOptimizer AdamOptimizer
      #  correctPrediction = tf.equal(tf.argmax(yOutput, 1), tf.argmax(self.yLabel, 1))
      #  accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))


