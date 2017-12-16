#coding=utf-8

import os
import random
import numpy as np
import tensorflow as tf

from dataPro import *
from baseDNNModel import *
from modelTrainer import *
from generateEggs import *

flags = tf.app.flags
flags.DEFINE_string("gpuId", "3", "Which gpu is assigned.")
flags.DEFINE_string("fileRootPath", "./files", "File path for all files.")
flags.DEFINE_string("dataRootPath", "./files", "Data file path for all data.")
flags.DEFINE_string("path4SaveEggsFile", "./files", "The path for saving eggs file.")
flags.DEFINE_string("path4Summaries", "./files/summaries", "The path for saving eggs file.")
flags.DEFINE_string("path4SaveModel", "./files/trainedModel", "The path for saving model.")
flags.DEFINE_string("oneCLDataPath4Training", "./files/4_training/PC3", "The path for training in one cell line.")
flags.DEFINE_string("oneCLDataPath4GenerateEggs", "./files/after_merge/PC3", "The path for generating eggs in one cell line.")

flags.DEFINE_float("testSize", 1e-1, "The threshold for validation data.")
flags.DEFINE_float("learningRate", 1e-4, "The learning rate for training.")
flags.DEFINE_float("dropOutRate", 0.5, "The threshold for validation data.")
flags.DEFINE_float("threshold4Val", 0.5, "The threshold for validation data.")
flags.DEFINE_float("threshold4Convegence", 1e-40, "The threshold for training convegence.")

flags.DEFINE_integer("batchSize", 150, "How many samples are trained in each iteration.")
flags.DEFINE_integer("trainEpoches", 1e3, "How many times training through all train data.")
flags.DEFINE_integer("nWeight", 10, "The weighted for negative samples in objective funcion.")
FLAGS = flags.FLAGS

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuId
  insDataPro = DataPro(FLAGS)
  insDataPro.loadDataInL1000()
  insDataPro.loadOneCLData()
  insDataPro.transferLabel2TwoCol()
  numOfNeurons = [1956, 200, 10, 2]
  insDNNModel = BaseDNNModel(FLAGS, numOfNeurons)
  insDNNModel.buildBaseDNNModelGraph()
  insModelTrainer = ModelTrainer(FLAGS, insDataPro, insDNNModel)
  modelSavePath = insModelTrainer.trainDNN()
  insGenerateEggs = GenerateEggs(FLAGS, insDataPro, modelSavePath)
  insGenerateEggs.generateEggs2Files()
  
