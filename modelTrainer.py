#coding=utf-8
import os
import math
import time
import random
import numpy as np
import tensorflow as tf

from dataSpliter import *
from resultStorer import *

class ModelTrainer:
  
  def __init__(self, FLAGS, insDataPro, insModel):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro
    self.insModel = insModel
    self.insDataSpliter = DataSpliter(FLAGS, insDataPro)
    self.insResultStorer = ResultStorer(FLAGS)

  # Training and validation for DNN
  def trainDNN(self):
    self.xTrain, self.xTest, self.yTrain, self.yTest = self.insDataSpliter.splitData2TrainAndVal()

    with tf.Session() as sess:
      oldTrainAccu, newTrainAccu, bestValAccu = 0.0, 0.0, 0.0
      flag = num4Epoches = 0

      saver = tf.train.Saver()
      init = tf.global_variables_initializer()
      sess.run(init)
      while True:
        trainIndex = np.array(range(self.xTrain.shape[0]))
        random.shuffle(trainIndex)
        print "No.%d epoch is starting..." % (num4Epoches)
        for ind in xrange(0, self.xTrain.shape[0], self.FLAGS.batchSize):
          batchXs, batchYs = self.xTrain[trainIndex[ind: ind + self.FLAGS.batchSize]], self.yTrain[trainIndex[ind: ind + self.FLAGS.batchSize]]
          #newTrainLoss, newTrainAccu, summary, tempTS = sess.run([self.insModel.loss, self.insModel.accuracy, self.insModel.merged, self.insModel.trainStep], feed_dict = {self.insModel.xData: batchXs, self.insModel.yLabel: batchYs, self.insModel.keepProb: self.FLAGS.dropOutRate})
          newTrainLoss, newTrainAccu, tempTS = sess.run([self.insModel.loss, self.insModel.accuracy, self.insModel.trainStep], feed_dict = {self.insModel.xData: batchXs, self.insModel.yLabel: batchYs, self.insModel.keepProb: self.FLAGS.dropOutRate})
          #ind4Summary = num4Epoches * math.ceil(self.xTrain.shape[0] * 1.0 / self.FLAGS.batchSize) + ind / self.FLAGS.batchSize
          #self.insModel.trainWriter.add_summary(summary, ind4Summary)
          self.insResultStorer.addLoss(newTrainLoss)
          self.insResultStorer.addTrainAccu(newTrainAccu)
          print "  The loss is %.6f. The training accuracy is %.6f..." % (newTrainLoss, newTrainAccu)

          if flag == 0:
            flag = 1
          else:
            if abs(newTrainAccu - oldTrainAccu) <= self.FLAGS.threshold4Convegence:
              flag = 2
          oldTrainAccu = newTrainAccu
        newValAccu = sess.run(self.insModel.accuracy, feed_dict = {self.insModel.xData: self.xTest, self.insModel.yLabel: self.yTest, self.insModel.keepProb: 1.0})
        #summary, newValAccu = sess.run([self.insModel.merged, self.insModel.accuracy], feed_dict = {self.insModel.xData: self.xTest, self.insModel.yLabel: self.yTest, self.insModel.keepProb: 1.0})
        #self.insModel.testWriter.add_summary(summary, num4Epoches)
        self.insResultStorer.addValAccu(newValAccu)
        print "    The validation accuracy is %.6f..." % (newValAccu)

        if newValAccu > bestValAccu:
          bestValAccu = newValAccu
          savePath = saver.save(sess, os.path.join(self.FLAGS.path4SaveModel, "model.ckpt"))
        if flag == 2 and num4Epoches >= self.FLAGS.trainEpoches:
          print "The training process is done..."
          print "The model saved in file:", savePath
          break
        num4Epoches += 1
    return savePath
