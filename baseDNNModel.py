#coding=utf-8
import os
import random
import numpy as np
import tensorflow as tf

from commonModelFunc import *

class BaseDNNModel(CommonModelFunc):
  
  def __init__(self, FLAGS, numOfNeurons):
    self.FLAGS = FLAGS
    self.numOfNeurons = np.array(numOfNeurons)
    self.numOfLayers = self.numOfNeurons.shape[0]

  # Building DNN graph for base model
  def buildBaseDNNModelGraph(self):
    with tf.device("/gpu:0"):
      self.keepProb = tf.placeholder(tf.float32, name = "keepProb")
      self.init = tf.global_variables_initializer()
      for ind, ele in enumerate(self.numOfNeurons):
        if ind == 0:  # Input layer
          self.xData = tf.placeholder(tf.float32, [None, ele], name = "xData")
        elif ind == self.numOfLayers - 1:  # Output layer
          self.yLabel = tf.placeholder(tf.float32, [None, ele], name = "yLabel")
          hHidden = tf.nn.dropout(hHidden, self.keepProb)
          name4VariableScope = "outputLayer"
          with tf.variable_scope(name4VariableScope):
            name4Weight, name4Bias, name4Act = "wOutput", "bOutput", "hOutput"
            wHidden = self.init_weight_variable(name4Weight, [self.numOfNeurons[ind - 1], ele])
            bHidden = self.init_bias_variable(name4Bias, [ele])
            self.hOutput = tf.matmul(hHidden, wHidden) + bHidden
            self.yOutput = tf.nn.softmax(self.hOutput, name = name4Act)
        else:  # Hidden layer
          name4VariableScope = "hidden" + str(ind) + "Layer"
          with tf.variable_scope(name4VariableScope):
            name4Weight, name4Bias, name4Act = "wHidden" + str(ind), "bHidden" + str(ind), "hHidden" + str(ind)
            wHidden = self.init_weight_variable(name4Weight, [self.numOfNeurons[ind - 1], ele])
            bHidden = self.init_bias_variable(name4Bias, [ele])
            if ind == 1:
              hHidden = tf.nn.relu(tf.matmul(self.xData, wHidden) + bHidden, name = name4Act)
            else:
              hHidden = tf.nn.relu(tf.matmul(hHidden, wHidden) + bHidden, name = name4Act)
      with tf.name_scope("lossAndAccuracy"):
        pItem = tf.reshape(tf.reduce_sum(self.yOutput, reduction_indices = [0]), [-1, 2])
        pMistakeItem = tf.reshape(tf.reduce_sum(self.yLabel * self.yOutput, reduction_indices = [0]), [-1, 2])
        nItem = tf.log(pItem)
        pRes = tf.matmul(pItem, tf.constant([[0.], [1.]]))
        nRes = tf.matmul(nItem, tf.constant([[1.], [0.]]))
        nMistakeRes = tf.matmul(pMistakeItem, tf.constant([[1.], [0.]]))
        self.loss = tf.subtract(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.hOutput, labels = self.yLabel)), self.FLAGS.nWeight * nMistakeRes, name = "loss")
        self.trainStep = tf.train.AdamOptimizer(self.FLAGS.learningRate).minimize(self.loss)
        correctPrediction = tf.equal(tf.argmax(self.yOutput, 1), tf.argmax(self.yLabel, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32), name = "accuracy")

