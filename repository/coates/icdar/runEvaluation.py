import numpy as np
import os
import sys
from scipy.io import loadmat
import cPickle as pickle
import coates
# init
if(len(sys.argv) > 1):
    coates.numOfClusters = int(sys.argv[1])
    pklFileName = sys.argv[2] 
else:
    pklFileName = 'segmentation_1.pkl'
print 'Number of clusters - ',coates.numOfClusters
coates.resultPath = os.path.join(coates.resultPath,str(coates.numOfClusters))
# load the dictionary
dictionaryPath = os.path.join(coates.resultPath, 'dictionary_' + str(coates.numOfClusters) + '.npy')
trainingParamsPath = os.path.join(coates.resultPath,'trainingParams.npz')
svmModelPath = os.path.join(coates.resultPath, 'svmModel.mat')
print 'Dictionary already generated. Hence Loading it from ', dictionaryPath
print 'Loading Training parameters from ', trainingParamsPath
print 'Loading the SVM Model - ', svmModelPath
# dictionary
finalDict = np.load(dictionaryPath)
trainingParamsList=[]
trainingParamsList.append(np.load(trainingParamsPath)['whitenTransform'])
trainingParamsList.append(np.load(trainingParamsPath)['meanValZCA'])
trainingParamsList.append(np.load(trainingParamsPath)['meanAvgValFeatures'])
trainingParamsList.append(np.load(trainingParamsPath)['stdAvgValFeatures'])
matContents = loadmat(svmModelPath)
pixelTextProbPredictorWeights = matContents['theta']
testPklFilePath = os.path.join(coates.testPath, pklFileName)
print 'Reading words pkl file from ', testPklFilePath
print '############################################################################################'
# store the results 
scalesFromMSER = False
# scales
scaleList = [0.1, 0.2 , 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
testPredictFolder = os.path.join(coates.resultPath, 'prediction')
if not os.path.exists(testPredictFolder):
    os.makedirs(testPredictFolder)
with open(testPklFilePath, 'rb') as pklFile:
    imgPathList = pickle.load(pklFile)
pklFile.close()

maxPredictFolder = os.path.join(testPredictFolder, 'maxPredict')
if not os.path.exists(maxPredictFolder):
    os.makedirs(maxPredictFolder)
if(scalesFromMSER == True):
    coates.getFinalPredictionMSER(finalDict, coates.cellSize, coates.stepSize,  coates.subCellSize,\
                    coates.subStepSize, trainingParamsList , maxPredictFolder,\
                    pixelTextProbPredictorWeights, imgPathList)
else:
    coates.getFinalPrediction(finalDict, coates.cellSize, coates.stepSize,  coates.subCellSize,\
                    coates.subStepSize, trainingParamsList , scaleList, maxPredictFolder,\
                    pixelTextProbPredictorWeights, imgPathList)
print '############################################################################################'
# exit the script
exit()