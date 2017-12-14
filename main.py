#coding=utf-8

import os
import random
import numpy as np
import tensorflow as tf

from dataPro import *
from baseDNNModel import *
from baseCNNModel import *
from modelTrainer import *
from generateEggs import *

flags = tf.app.flags
flags.DEFINE_string("gpuId", "0", "Which gpu is assigned.")
flags.DEFINE_string("fileRootPath", "/home/xlw/second", "File path for all files.")
flags.DEFINE_string("dataRootPath", "/home/xlw/second", "Data file path for all data.")
flags.DEFINE_string("path4SaveModel", "/home/xlw/second/DNN-eggs/trainedModel", "The path for saving model.")
flags.DEFINE_string("path4SaveEggsFile", "/home/xlw/second/DNN-eggs/eggfile/", "The path for saving eggs file.")
flags.DEFINE_string("savePath", "/home/xlw/second/CNN_ensemble/1_fold/", "The path for saving loss and accuracy.")
flags.DEFINE_string("ensembleDataPath", "/home/xlw/second/CNN_ensemble/4training_ensemble/", "The path for enmsemble data.")
flags.DEFINE_string("oneCLDataPath4Training", "/home/xlw/second/DNN-eggs/4_training/PC3", "The path for training in one cell line.")
flags.DEFINE_string("oneCLDataPath4GenerateEggs", "/home/xlw/second/DNN-eggs/after_merge/PC3", "The path for generating eggs in one cell line.")

flags.DEFINE_float("testSize", 0.1, "The threshold for validation data.")
flags.DEFINE_float("dropOutRate", 0.5, "The threshold for validation data.")
flags.DEFINE_float("learningRate", 0.0001, "The learning rate for training.")
flags.DEFINE_float("threshold4Val", 0.5, "The threshold for validation data.")
flags.DEFINE_float("threshold4Convegence", 1e-40, "The threshold for training convegence.")

flags.DEFINE_integer("num4FC", 10, "The number of neurons in FC layer.")
flags.DEFINE_integer("conv1KWidth", 10, "The width of convolutional kernel.")
flags.DEFINE_integer("conv1SWidth", 1, "The width of convolutional kernel stride.")
flags.DEFINE_integer("maxInputChannels", 7, "The maximum number of input channels.")
flags.DEFINE_integer("batchSize", 150, "How many samples are trained in each iteration.")
flags.DEFINE_integer("trainEpoches", 1000, "How many times training through all train data.")
flags.DEFINE_integer("nWeight", 10, "The weighted for negative samples in objective funcion.")
flags.DEFINE_integer("num4OutputChannels", 10, "The number of output channels in first convolutional layer.")
FLAGS = flags.FLAGS

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuId
  insDataPro = DataPro(FLAGS)
  # For DNN
  insDataPro.loadDataInL1000()
  #print insDataPro.drugSampleId.shape
  #print insDataPro.drugData.shape
  #print insDataPro.tpSampleId.shape
  #print insDataPro.tpData.shape
  #print insDataPro.dnaIdNew.shape
  #print insDataPro.yLabel.shape
  #print insDataPro.realNegativeTpData.shape
  #print "All done..."
  #---------------------------------
  insDataPro.loadOneCLData()
  insDataPro.transferLabel2TwoCol()
  numOfNeurons = [1956, 200, 10, 2]
  insDNNModel = BaseDNNModel(FLAGS, numOfNeurons)
  insDNNModel.buildBaseDNNModelGraph()
  insModelTrainer = ModelTrainer(FLAGS, insDataPro, insDNNModel)
  modelSavePath = insModelTrainer.trainDNN()
  insGenerateEggs = GenerateEggs(FLAGS, insDataPro, modelSavePath)
  insGenerateEggs.generateEggs2Files()

  # For CNN
  #insDataPro.loadEnsembleDataAndLabel(FLAGS.ensembleDataPath)
  #print insDataPro.positiveData.shape, insDataPro.negativeData.shape, insDataPro.allTrainData.shape
  #print insDataPro.allTrainLabel.shape
  #insCNNModel = BaseCNNModel(FLAGS, insDataPro)
  #insCNNModel.buildBaseCNNModelGraph()
