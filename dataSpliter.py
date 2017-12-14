#coding=utf-8
import numpy as np
from sklearn.model_selection import train_test_split

class DataSpliter:

  def __init__(self, FLAGS, insDataPro):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro

  def splitData2TrainAndVal(self):
    xTrain, xTest, yTrain, yTest = train_test_split(self.insDataPro.allTrainData, self.insDataPro.allTrainLabel, test_size = self.FLAGS.testSize, random_state = 42)
    return xTrain, xTest, yTrain, yTest
