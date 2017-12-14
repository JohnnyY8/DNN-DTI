import os
import numpy as np
import tensorflow as tf

class GenerateEggs():
  
  def __init__(self, FLAGS, insDataPro, modelSavePath):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro
    self.modelSavePath = modelSavePath

  def generateEggs2Files(self):
    self.insDataPro.generateUnlabeledData4Eggs()
    tf.reset_default_graph()
    with tf.Session() as sess:
      saver = tf.train.import_meta_graph(self.modelSavePath + ".meta")
      graph = tf.get_default_graph()
      saver.restore(sess, self.modelSavePath)
      xData = graph.get_operation_by_name("xData").outputs[0]
      yLabel = graph.get_operation_by_name("yLabel").outputs[0]
      yOutput = graph.get_operation_by_name("outputLayer/hOutput").outputs[0]
      keepProb = graph.get_operation_by_name("keepProb").outputs[0]
      for i in xrange(0, self.insDataPro.allUnlabeledData.shape[0], self.FLAGS.batchSize):
        feedData = {xData: self.insDataPro.allUnlabeledData[i: i + self.FLAGS.batchSize], yLabel: np.zeros((self.FLAGS.batchSize, 2)), keepProb: 1.0}
        probTemp = sess.run(yOutput, feed_dict = feedData)
        if i == 0:
          probRes = probTemp
        else:
          probRes = np.append(probRes, probTemp, axis = 0)
      print "self.insDataPro.allUnlabeledData:", self.insDataPro.allUnlabeledData.shape
      print "probRes.shape:", probRes.shape
      #print "probRes:", probRes
      #print sum(probRes[0])
      #print sum(probRes[1])
      #print sum(probRes[2])
    
    with open(os.path.join(self.FLAGS.path4SaveEggsFile, "eggsfile.txt"), 'w') as filePointer:
      # Write drug names
      flag = 0
      for iele in self.insDataPro.drugName:
        if flag == 0:
          strLine = str(iele)
          flag = 1
        else:
          strLine += '\t' + str(iele)
      strLine += '\n'
      filePointer.write(strLine)
      # Write tp names and distance
      ind4dis = 0
      distance = self.insDataPro.calcDistance(probRes)
      print "distance.shape:", distance.shape
      print "self.insDataPro.mapMatrix.shape:", self.insDataPro.mapMatrix.shape, np.sum(self.insDataPro.mapMatrix)
      count4Eggs = 0.0
      for ind, iele in enumerate(self.insDataPro.tpName):
        strLine = str(iele)
        for jele in self.insDataPro.mapMatrix[ind]:
          if jele < 0.5:
            strLine += '\t' + str(distance[ind4dis][0])
            if distance[ind4dis][0] > 0:
              count4Eggs += 1
            ind4dis += 1
          elif jele > 0.5:
            strLine += '\t' + "6666666"
        strLine += '\n'
        filePointer.write(strLine)
      print "The rate of eggs is:", count4Eggs / self.insDataPro.allUnlabeledData.shape[0]
