import numpy as np
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import savemat, loadmat
from subprocess import call
import coates
import helper
####################################################################################################
# 0- file name
# 1 - number of clusters
if(len(sys.argv) > 1):
    coates.numOfClusters = int(sys.argv[1])
# make the necessary directory
coates.resultPath = os.path.join(coates.resultPath,str(coates.numOfClusters))
if not os.path.exists(coates.resultPath):
    os.makedirs(coates.resultPath)    
print 'Number of clusters - ',coates.numOfClusters 
###################################################################################################
# necssary path variables
dictionaryPath = os.path.join(coates.resultPath, 'dictionary_' + str(coates.numOfClusters) + '.npy')
trainingParamsPath = os.path.join(coates.resultPath,'trainingParams.npz')
tData_idfier = str(coates.cellSize[0]) + '_' + str(coates.cellSize[1])
dictData_idfier = str(coates.subCellSize[0]) + '_' + str(coates.subCellSize[1])
# dictionary data points save location
dictDataPointsPath = os.path.join(coates.trainPath, 'dictDataPoints_' + dictData_idfier +'.npy')  
# positive and negative patch save location
positivePatchesSavePath = os.path.join(coates.trainPath, 'positivePatches_'+ tData_idfier + '.npy')
negativePatchesSavePath = os.path.join(coates.trainPath, 'negativePatches_'+ tData_idfier + '.npy')
positivePatchesTestSavePath = os.path.join(coates.trainPath, 'positivePatchesTest_'+ tData_idfier + '.npy')
negativePatchesTestSavePath = os.path.join(coates.trainPath, 'negativePatchesTest_'+ tData_idfier + '.npy')

feStandardize = True
if(os.path.isfile(dictionaryPath)):
    print 'Dictionary already generated. Hence Loading it from ', dictionaryPath
    finalDict = np.load(dictionaryPath) 
    print 'Shape of the dictionary - ', finalDict.shape  
    # load the training params obtained during training
    print 'Loading Training parameters from ', trainingParamsPath
    trainingParamsList=[]
    trainingParamsList.append(np.load(trainingParamsPath)['whitenTransform'])
    trainingParamsList.append(np.load(trainingParamsPath)['meanValZCA'])
else:    
    print 'Generating dictionary'
    print 'Loading the data from ', dictDataPointsPath
    dictDataPoints = np.load(dictDataPointsPath)         
    # Note - Array shape  - Number of Data points x Dimension
    print 'Dimension - ', dictDataPoints.shape[1]
    print 'Number of Data Points - ', dictDataPoints.shape[0] 
       
    # save the figure of data points
    numSelect = 64
    randomSelect = np.random.randint(0, dictDataPoints.shape[0], numSelect)
    imgs = np.reshape(dictDataPoints[randomSelect],(numSelect, coates.subCellSize[0], \
                                                        coates.subCellSize[1]))
    helper.displayImgs(imgs,(8, 8))  
    plt.savefig(os.path.join(coates.resultPath, 'rawDataPoints.png'))
    
    # prepare the inputs for training - 
    dictDataPoints, trainingParamsList =  coates.prepareData(dictDataPoints, coates.constantZCA)
    
    print 'Covariance matrix of whitened data'
    covWhitenData =  np.dot(dictDataPoints.T,dictDataPoints) / dictDataPoints.shape[0]
    print  np.diag(covWhitenData)
    
    plt.clf()    
    plt.imshow(covWhitenData, cmap = cm.get_cmap('ocean'))
    plt.savefig(os.path.join(coates.resultPath, 'covarianceWhitenedData.png'))
    
    # save the figure of whitened data points
    imgs = np.reshape(dictDataPoints[randomSelect],(numSelect, coates.subCellSize[0], \
                                                        coates.subCellSize[1]))
    helper.displayImgs(imgs,(8, 8))  
    plt.savefig(os.path.join(coates.resultPath, 'whitenedDataPoints.png'))
   
    # save the eigen value graph
    plt.clf()
    eigValues = np.sort(abs(np.linalg.eig(covWhitenData)[0]))
    plt.scatter(np.arange(eigValues.shape[0]),eigValues)
    plt.savefig(os.path.join(coates.resultPath, 'eigenValuesWhitenedData.png'))
       
    print 'Number of params in the List - ', len(trainingParamsList)
    print 'Saving Training parameters'
    np.savez_compressed(trainingParamsPath, whitenTransform = trainingParamsList[0],\
                         meanValZCA = trainingParamsList[1])
        
    # input to sherical kmeans
    finalDict = coates.sphericalKmeans(dictDataPoints, \
                                       coates.numOfClusters, coates.numIterations) 
       
    print 'Dictionary Shape - ', finalDict.shape
    np.save(dictionaryPath, finalDict)     
    numCenteroids = finalDict.shape[0]
    
    # saving the figures
    imgs = np.reshape(finalDict,(numCenteroids, coates.subCellSize[0], \
                                   coates.subCellSize[1]))
    if coates.numOfClusters == 32:
        layout = (8, 4)
    elif coates.numOfClusters == 64:
        layout = (8, 8)
    elif coates.numOfClusters == 200:
        layout = (20, 10)
    elif coates.numOfClusters == 500:
        layout = (25, 20)
    elif coates.numOfClusters == 1000:
        layout = (40, 25) 
                        
    helper.displayImgs(imgs, layout)  
    plt.savefig(os.path.join(coates.resultPath, 'dictionary_' + str(coates.numOfClusters) + '.png'))
    # memory clean up
    del dictDataPoints
print '############################################################################################'    
###################################################################################################
# training
trainDataPath = os.path.join(coates.resultPath, 'trainingSet.npy')
testDataPath = os.path.join(coates.resultPath, 'testDataSet.npy')
# for matlab 
trainDataPathMatFile = os.path.join(coates.resultPath, 'trainingSetMatlab.mat')
testDataPathMatFiles = os.path.join(coates.resultPath, 'testDataSetMatlab.mat')
if(os.path.isfile(trainDataPath)):
    print 'Training Data already generated.Loading from ', trainDataPath
    trainDataSet = np.load(trainDataPath)
    print 'Shape of the Training data - ', trainDataSet.shape
    print 'Test Data already generated.Loading from ', testDataPath
    testDataSet = np.load(testDataPath)
    print 'Shape of the Test data - ', testDataSet.shape
    feStandardized = []
    feStandardized.append(np.load(trainingParamsPath)['meanAvgValFeatures'])
    feStandardized.append(np.load(trainingParamsPath)['stdAvgValFeatures'])
else:
    print 'Generating Training Data.'
    print 'Loading - ',positivePatchesSavePath
    print 'Loading - ',negativePatchesSavePath
    positivePatches = np.load(positivePatchesSavePath)
    negativePatches = np.load(negativePatchesSavePath)[0:30000,:,:]
    print 'Text Patches - ', positivePatches.shape
    print 'Non Text Patches - ', negativePatches.shape
    ################################################################################################
    # Now we have generated the required number of text patches and non text patches
    # Generating features
    # consists of preprocessing ,ZCA whitening, Average pooling operations
    # patch extraction
    positivePatches = coates.computeResponsePatches(positivePatches, finalDict, coates.cellSize,\
                                       coates.subCellSize, coates.subStepSize, \
                                       trainingParamsList, 1)
    negativePatches = coates.computeResponsePatches(negativePatches, finalDict, coates.cellSize,\
                                           coates.subCellSize, coates.subStepSize, \
                                           trainingParamsList, 1) 
    # stack up ...to build the training data
    # training data shape - Number of samples X features dimension
    # save the taining set
    # training data shape - Number of samples X features dimension
    trainDataSet = np.vstack((positivePatches, negativePatches))
    
    if(feStandardize):
        trainDataSet, feStandardized = coates.standardizeFeatures(trainDataSet)
    else:
        feStandardized = [0,1]
             
    outputlabels = np.vstack((np.ones((positivePatches.shape[0],1)), 2*np.ones((negativePatches.shape[0],1))))
    trainDataSet = np.hstack((outputlabels, trainDataSet))
    print 'Mean (training data) - ', np.mean(trainDataSet[:,1:])
    print 'Std Dev (training data)- ', np.std(trainDataSet[:,1:])
        
    if(np.isnan(trainDataSet).sum()):
        print 'Warning - NaNs in the training data'
    print 'Saving training data'
    print 'Positive training examples - ',positivePatches.shape
    print 'Negative training examples - ',negativePatches.shape
    print 'Training Data Shape - ', trainDataSet.shape
    # add intercept
    trainDataSet = np.hstack((trainDataSet, np.ones((trainDataSet.shape[0],1))))
    np.save(trainDataPath, trainDataSet)
    # matlab
    savemat(trainDataPathMatFile, {'trainData':trainDataSet})
    # for test
    print 'Generating Testing Data.'
    print 'Loading - ',positivePatchesTestSavePath
    print 'Loading - ',negativePatchesTestSavePath
    positivePatchesTest = np.load(positivePatchesTestSavePath)
    negativePatchesTest = np.load(negativePatchesTestSavePath)
    print 'Text Patches - ', positivePatchesTest.shape
    print 'Non Text Patches - ', negativePatchesTest.shape
    ################################################################################################
    # Generating features
    # consists of preprocessing ,ZCA whitening, Average pooling operations    
    # patch extraction
    positivePatchesTest = coates.computeResponsePatches(positivePatchesTest, finalDict, coates.cellSize,\
                                           coates.subCellSize, coates.subStepSize, \
                                           trainingParamsList, 1)
    negativePatchesTest = coates.computeResponsePatches(negativePatchesTest, finalDict, coates.cellSize,\
                                           coates.subCellSize, coates.subStepSize, \
                                           trainingParamsList, 1) 
    ################################################################################################
    positivePatchesTest = coates.standardizeFeatures(positivePatchesTest,feStandardized)[0]
    negativePatchesTest = coates.standardizeFeatures(negativePatchesTest,feStandardized)[0]
    positivePatchesTest = np.hstack((np.ones((positivePatchesTest.shape[0],1)), positivePatchesTest))
    negativePatchesTest = np.hstack((np.ones((negativePatchesTest.shape[0],1))*2, negativePatchesTest))
    # save the taining set
    # training data shape - Number of samples X features dimension
    testDataSet = np.vstack((positivePatchesTest, negativePatchesTest))
    print 'Mean (test data) - ', np.mean(testDataSet[:,1:])
    print 'Std Dev (test data)- ', np.std(testDataSet[:,1:]) 
    print 'Test Data Shape - ', testDataSet.shape
    # add intercept
    testDataSet = np.hstack((testDataSet, np.ones((testDataSet.shape[0],1))))
    np.save(testDataPath, testDataSet)
    # matlab
    savemat(testDataPathMatFiles, {'testData':testDataSet}) 
    # save the values
    np.savez_compressed(trainingParamsPath, whitenTransform = trainingParamsList[0],\
                        meanValZCA = trainingParamsList[1], \
                        meanAvgValFeatures = feStandardized[0],\
                        stdAvgValFeatures = feStandardized[1])
    # memory clean up
    del negativePatchesTest
    del positivePatchesTest
    del negativePatches
    del positivePatches
print '############################################################################################'
###################################################################################################
# training the SVM ...done in matlab
svmModelPath = os.path.join(coates.resultPath, 'svmModel.mat')
if(os.path.isfile(svmModelPath)):
    print 'SVM Model already generated'
    print 'Loading the SVM Model - ', svmModelPath
    matContents = loadmat(svmModelPath)
    pixelTextProbPredictorWeights = matContents['theta']
    svmStats = matContents['svmStats']
    print 'Selected Model lambda value - ', svmStats[0,0], ' C - ', 1/svmStats[0,0]
    print 'Training accuracy - ', svmStats[0,1]
    print 'Testing accuracy - ', svmStats[0,2]
else:    
    call(["/usr/local/MATLAB/R2013a/bin/matlab -nosplash -nodesktop -r \"runSVMTrainGetModel(\'%s\')\"" %(coates.resultPath)], shell=True)
    # machine pool path - /vol/local/amd64/matlab2014a/bin 
    #call(["/vol/local/amd64/matlab2014a/bin/matlab -nosplash -nodesktop -r \"runSVMTrainGetModel(\'%s\')\"" %(coates.resultPath)], shell=True)
    time.sleep(30)
    print 'Loading the SVM Model - ', svmModelPath
    matContents = loadmat(svmModelPath)
    pixelTextProbPredictorWeights = matContents['theta']
    svmStats = matContents['svmStats']
    print 'Selected Model lambda value - ', svmStats[0,0], ' C - ', 1/svmStats[0,0]
    print 'Training accuracy - ', svmStats[0,1]
    print 'Testing accuracy - ', svmStats[0,2]   
print '############################################################################################'
# exit the script
exit()