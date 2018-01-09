#coding=utf-8
import os
import numpy as np
import tensorflow as tf

class DataPro:

  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    print("1.PC3; 2.VCAP; 3.A375; 4.A549; 5.HA1E; 6.HCC515; 7.HEPG2;")
    #numCl = input("Please choose the number for cell line: ")
    numCl = 1
    dictCl = {
        1: "PC3",
        2: "VCAP",
        3: "A375",
        4: "A549",
        5: "HA1E",
        6: "HCC515",
        7: "HEPG2"}
    self.clName = dictCl[numCl]
    self.fileRootPath = FLAGS.fileRootPath
    self.dateRootPath = FLAGS.dataRootPath
    self.cpShPath = os.path.join(
        FLAGS.dataRootPath,
        "drug_data/cp_sh",
        "cp_sh_map_" + self.clName + ".txt")
    self.trtCpPath = os.path.join(
        FLAGS.dataRootPath,
        "trt_cp_" + self.clName + ".txt")
    self.trtShPath = os.path.join(
        FLAGS.dataRootPath,
        "trt_sh_" + self.clName + ".txt")
    self.dnaIdPath = os.path.join(
        FLAGS.dataRootPath,
        "dna_id_file.txt")
    self.mergeSameDrugMapFilePath = os.path.join(
        FLAGS.fileRootPath,
        "trt_cp.info")
    self.L1000GeneIdFilePath = os.path.join(
        FLAGS.fileRootPath,
        "Gene_ID.txt")
    self.L1000MapFilePath = os.path.join(
        FLAGS.fileRootPath,
        "GPL96.annot")

  # Get drug sample id and drug data
  def getDrugSampleIdAndDrugData(self):
    with open(self.trtCpPath) as filePointer:
      fileLines = filePointer.readlines()
    fileLines = [fileLine[: -1].split('\t') for fileLine in fileLines]
    self.drugSampleId = np.array(fileLines[0])
    self.drugData = np.array(fileLines[1: ])

  # Get tp sample id and tp data
  def getTpSampleIdAndTpData(self):
    with open(self.trtShPath) as filePointer:
      fileLines = filePointer.readlines()
    fileLines = [fileLine[: -1].split('\t') for fileLine in fileLines]
    self.tpSampleId = np.array(fileLines[0])
    self.tpData = np.array(fileLines[1: ])

  # Get dna id
  def getDnaId(self):
    with open(self.dnaIdPath) as filePointer:
      fileLines = filePointer.readlines()
    self.dnaId = np.array(fileLines[0].split('\t'))

  # Get ylabel
  def getyLabel(self):
    with open(self.cpShPath) as filePointer:
      fileLines = filePointer.readlines()
    fileLines = [fileLine[: -1].split('\t') for fileLine in fileLines]
    self.yLabel = np.array(
        [fileLine[1: ] for fileLine in fileLines[1: ]],
        dtype = "float")

  # Get real negative tp data
  def getRealNegativeTpData(self):
    realNegativeTpFilePath = os.path.join(
        self.dateRootPath,
        "real_negative",
        "tp_data_" + self.clName + "_978.npy")
    self.realNegativeTpData = np.load(realNegativeTpFilePath)

  # Get all data in 22268 dimensions
  def loadAllData(self):
    self.getDrugSampleIdAndDrugData()
    self.getTpSampleIdAndTpData()
    self.getDnaId()
    self.getyLabel()
    self.getRealNegativeTpData()
    print("Load all data in 22268 dimensions is done...")

  # Merge data from the same drug
  def mergeDataFromSameDrug():
    sampleIdNew, xDataNew = [], []
    # Create dict with the key of drug and the value of samples
    #     in this cell line
    dictDrugSample = {}
    with open(self.mergeSameDrugMapFilePath) as mapFile:
      fileLines = mapFile.readlines()
    fileLines = [fileLine.split('\t') for fileLine in fileLines]
    for fileLine in fileLines:
      mapKey = fileLine[1]
      if not dictDrugSample.has_key(mapKey):
        dictDrugSample[mapKey] = []
      mapValue = fileLine[0]
      if mapValue in self.drugSampleId:
        dictDrugSample[mapKey].append(mapValue)
    # Merge tha samples from same drug
    for eachKey in dictDrugSample:
      eachValue = dictDrugSample[eachKey]
      if len(eachValue) == 0:
        continue
      sampleIdNew.append(eachKey)
      for i, iele in enumerate(eachValue):
        if i == 0:
          eachxDataNew = np.array(
              self.drugData[np.where(self.drugSampleId == iele)[0][0]])
        else:
          eachxDataNew += np.array(
              self.drugData[np.where(self.drugSampleId == iele)[0][0]])
      xDataNew.append(list(eachxDataNew / len(eachValue)))
    self.drugSampleId = np.array(sampleIdNew)
    self.drugData = np.array(xDataNew)
    print("Merging data from same drug is done...")

  # Merge L1000 dna id according to all gene id
  def mergeDnaId(self, xData):
    dnaIdNew, xDataNew = [], []
    # Load L1000 gene id
    with open(self.L1000GeneIdFilePath) as geneIdFile:
      fileLines = geneIdFile.readlines()
    del fileLines[0]
    L1000GeneId = [fileLine[: -2] for fileLine in fileLines]
    # Load mapping file of L1000 gene id and dna id
    with open(self.L1000MapFilePath) as L1000MapFile:
      fileLines = L1000MapFile.readlines()
    # Remove all useless lines
    for i in range(28):
      del fileLines[0]
    del fileLines[len(fileLines) - 1]
    # Construct dictionary with the key of geneid and the value of dnaid
    dictL1000GeneIdDnaId = {}
    listLines = [fileLine.split('\t') for fileLine in fileLines]
    for listLine in listLines:
      if listLine[3] in L1000GeneId:
        if not dictL1000GeneIdDnaId.has_key(listLine[3]):
          dictL1000GeneIdDnaId[listLine[3]] = [
              np.where(self.dnaId == listLine[0])[0][0]]
        else:
          dictL1000GeneIdDnaId[listLine[3]].append(
              np.where(self.dnaId == listLine[0])[0][0])
    for eachKey in dictL1000GeneIdDnaId:
      dnaIdNew.append(eachKey)
    # Merge dna id according to gene id
    for eachxData in xData:
      eachxDataNew = []
      for eachKey in dictL1000GeneIdDnaId:
        listValue = dictL1000GeneIdDnaId[eachKey]
        sumFeature = 0.0
        for eachValue in listValue:
          sumFeature += float(eachxData[eachValue])
        newFeature = sumFeature / len(listValue)
        eachxDataNew.append(newFeature)
      xDataNew.append(eachxDataNew)
    self.dnaIdNew = np.array(dnaIdNew)
    return np.array(xDataNew)

  # Get all data in L1000 manner
  def loadDataInL1000(self):
    self.loadAllData()
    self.drugData = self.mergeDnaId(self.drugData)
    self.tpData = self.mergeDnaId(self.tpData)
    print("Load all data in L1000 manner is done...")

  # Load preprocessed data for training and validation
  def loadDataAndLabel(self, basePath4DataAndLabel):
    self.allTrainData = np.load(
        os.path.join(
            basePath4DataAndLabel,
            "allTrainDataFor5Cl.npy"))
    self.allTrainLabel = np.load(
        os.path.join(
            basePath4DataAndLabel,
            "allTrainLabelFor5Cl.npy"))
    self.allValData = np.load(
        os.path.join(
            basePath4DataAndLabel,
            "allValDataFor5Cl.npy"))
    self.allValLabel = np.load(
        os.path.join(
            basePath4DataAndLabel,
            "allValLabelFor5Cl.npy"))
    print("Load preprocessed data for training and validation is done...")

  # Load one cell line data
  def loadOneCLData(self):
    self.allTrainData = np.load(
        os.path.join(
            self.FLAGS.oneCLDataPath4Training,
            "x_data.npy"))
    self.allTrainLabel = np.load(
        os.path.join(
            self.FLAGS.oneCLDataPath4Training,
            "y_label.npy"))
    print("Load %s cell line data is done..." % (self.clName))

  # Transfer label from one col to two cols
  def transferLabel2TwoCol(self):
    firstRow = self.allTrainLabel.reshape(-1)
    secondRow = np.ones(firstRow.shape) - firstRow
    self.allTrainLabel = np.vstack((secondRow, firstRow)).transpose()
    print("Transfer label from one col to two cols is done...")

  # Generate unlabeled data for eggs
  def generateUnlabeledData4Eggs(self):
    self.tpName = np.load(
        os.path.join(
            self.FLAGS.oneCLDataPath4GenerateEggs,
            "tp_name.npy"))
    tpData = np.load(
        os.path.join(
            self.FLAGS.oneCLDataPath4GenerateEggs,
            "tp_data.npy"))
    self.drugName = np.load(
        os.path.join(
            self.FLAGS.oneCLDataPath4GenerateEggs,
            "drug_name.npy"))
    drugData = np.load(
        os.path.join(
            self.FLAGS.oneCLDataPath4GenerateEggs,
            "drug_data.npy"))
    self.mapMatrix = np.load(
        os.path.join(
            self.FLAGS.oneCLDataPath4GenerateEggs,
            "new_label.npy"))

    nonZeroIndex = np.where(self.mapMatrix == 0)
    self.allUnlabeledData = np.hstack(
        (self.tpData[nonZeroIndex[0]], self.drugData[nonZeroIndex[1]]))
    print("Generate unlabeled data is done...")

  # Calc all distance
  def calcDistance(self, prob):
    distance = np.log(prob / (1 - prob))
    return distance[:, [1]]

  # Load ensemble data
  def loadEnsembleDataAndLabel(self, ensembleDataPath):
    self.positiveData = np.load(
        os.path.join(
            ensembleDataPath,
            "positiveDataAppendZeros.npy"))
    self.negativeData = np.load(
        os.path.join(
            ensembleDataPath,
            "negativeDataAppendZeros.npy"))

    self.allTrainData = np.vstack((self.positiveData, self.negativeData))
    self.buildEnsembleLabel()
    print("Load ensemble data and label is done...")

  # Build ensemble label
  def buildEnsembleLabel(self):
    positiveLabel, negativeLabel = \
        np.ones(self.positiveData.shape[0]), \
        np.zeros(self.negativeData.shape[0])

    self.allTrainLabel = np.append(positiveLabel, negativeLabel)
    self.transferLabel2TwoCol()
    print("Build ensemble label is done...")
