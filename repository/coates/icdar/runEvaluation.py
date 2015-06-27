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
saveIntermediateResults = False 
scalesFromMSER = False

# scales
scaleList = [0.1, 0.2 , 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
testPredictFolder = os.path.join(coates.resultPath, 'prediction')
if not os.path.exists(testPredictFolder):
    os.makedirs(testPredictFolder)
with open(testPklFilePath, 'rb') as pklFile:
    imgPathList = pickle.load(pklFile)
pklFile.close()

imgPathList = imgPathList[:1]
imgPathList[0] = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','test','ryoungt_05.08.2002','aPICT0035.JPG')
#imgPathList[0] = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','test','ryoungt_13.08.2002','dPICT0020.JPG')
#imgPathList[0] = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','test','sml_01.08.2002','IMG_1200.JPG')

if saveIntermediateResults == True:
    coates.parallelImageProcessing(finalDict, coates.cellSize, coates.stepSize,  coates.subCellSize,\
                        coates.subStepSize, trainingParamsList , scaleList, testPredictFolder,\
                        pixelTextProbPredictorWeights, imgPathList)
    # folder to store the max prediction values
    print 'Generating final predictions'
    maxPredictFolder = os.path.join(testPredictFolder, 'maxPredict')
    gTruthFolder = os.path.join(coates.testPath, 'groundTruth')
    if not os.path.exists(maxPredictFolder):
        os.makedirs(maxPredictFolder)
    # calculate max image
    coates.findMaxImage(gTruthFolder,testPredictFolder, maxPredictFolder, scaleList, coates.stepSize, \
                                    coates.cellSize, imgPathList)
else:
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