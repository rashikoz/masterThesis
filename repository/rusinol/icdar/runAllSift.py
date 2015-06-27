import numpy as np
import os
import time
import rusinol
import xml.etree.ElementTree as xmlParser
from scipy.io import savemat, loadmat
from sklearn.externals import joblib
from scipy.misc import imresize,imread
from subprocess import call
import cPickle as pickle
import vlfeat
import mserHelper
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2 as cv
###################################################################################################
# necssary path variables
tData_idfier = str(rusinol.cellSize[0]) + '_' + str(rusinol.cellSize[1])  
# positive and negative patch save location
positivePatchesSavePath = os.path.join(rusinol.trainPath, 'positivePatches_'+ tData_idfier + '.npy')
negativePatchesSavePath = os.path.join(rusinol.trainPath, 'negativePatches_'+ tData_idfier + '.npy')
positivePatchesTestSavePath = os.path.join(rusinol.trainPath, 'positivePatchesTest_'+ tData_idfier + '.npy')
negativePatchesTestSavePath = os.path.join(rusinol.trainPath, 'negativePatchesTest_'+ tData_idfier + '.npy')
# training
trainDataPath = os.path.join(rusinol.resultPath, 'trainingSet.npy')
testDataPath = os.path.join(rusinol.resultPath, 'testDataSet.npy')
trainDataPathMatFile = os.path.join(rusinol.resultPath, 'trainingSetMatlab.mat')
testDataPathMatFiles = os.path.join(rusinol.resultPath, 'testDataSetMatlab.mat')
trainingParamsPath = os.path.join(rusinol.resultPath,'trainingParams.npz')
feStandardize = True
scalesFromMSER = False
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
    negativePatches = np.load(negativePatchesSavePath)
    print 'Text Patches - ', positivePatches.shape
    print 'Non Text Patches - ', negativePatches.shape
    # Now we have generated the required number of text patches and non text patches
    # Generating SIFT features
    patchSize = (positivePatches.shape[1],positivePatches.shape[2])
    positivePatches = rusinol.getSIFTDescriptor(positivePatches, patchSize)
    # Generating features
    patchSize = (negativePatches.shape[1],negativePatches.shape[2])
    negativePatches = rusinol.getSIFTDescriptor(negativePatches, patchSize)
    
    trainDataSet = np.vstack((positivePatches, negativePatches))
    if(feStandardize):
        trainDataSet, feStandardized = rusinol.standardizeFeatures(trainDataSet)
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
    # add intercept     # matlab
    trainDataSet = np.hstack((trainDataSet, np.ones((trainDataSet.shape[0],1))))
    np.save(trainDataPath, trainDataSet)
    savemat(trainDataPathMatFile, {'trainData':trainDataSet})
    # test
    print 'Generating Testing Data.'
    print 'Loading - ',positivePatchesTestSavePath
    print 'Loading - ',negativePatchesTestSavePath
    positivePatchesTest = np.load(positivePatchesTestSavePath)
    negativePatchesTest = np.load(negativePatchesTestSavePath)
    print 'Text Patches - ', positivePatchesTest.shape
    print 'Non Text Patches - ', negativePatchesTest.shape
    
    patchSize = (positivePatchesTest.shape[1],positivePatchesTest.shape[2])
    positivePatchesTest = rusinol.getSIFTDescriptor(positivePatchesTest, patchSize)
    # Generating features
    patchSize = (negativePatchesTest.shape[1],negativePatchesTest.shape[2])
    negativePatchesTest = rusinol.getSIFTDescriptor(negativePatchesTest, patchSize)
    # standardize
    positivePatchesTest = rusinol.standardizeFeatures(positivePatchesTest,feStandardized)[0]
    negativePatchesTest = rusinol.standardizeFeatures(negativePatchesTest,feStandardized)[0]
    positivePatchesTest = np.hstack((np.ones((positivePatchesTest.shape[0],1)), positivePatchesTest))
    negativePatchesTest = np.hstack((2*np.ones((negativePatchesTest.shape[0],1)), negativePatchesTest))
    
    # save the taining set
    # training data shape - Number of samples X features dimension
    testDataSet = np.vstack((positivePatchesTest, negativePatchesTest))
    print 'Mean (test data) - ', np.mean(testDataSet[:,1:])
    print 'Std Dev (test data)- ', np.std(testDataSet[:,1:]) 
    print 'Test Data Shape - ', testDataSet.shape
    # add intercept     # matlab
    testDataSet = np.hstack((testDataSet, np.ones((testDataSet.shape[0],1))))
    np.save(testDataPath, testDataSet)
    savemat(testDataPathMatFiles, {'testData':testDataSet}) 
    # save the values
    np.savez_compressed(trainingParamsPath, meanAvgValFeatures = feStandardized[0],\
                        stdAvgValFeatures = feStandardized[1])
    # memory clean up
    del negativePatchesTest
    del positivePatchesTest
    del negativePatches
    del positivePatches
print '############################################################################################'
###################################################################################################
# training the SVM ...done in matlab
svmModelPath = os.path.join(rusinol.resultPath, 'svmModel.mat')
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
    call(["/usr/local/MATLAB/R2013a/bin/matlab -nosplash -nodesktop -r \"runSVMTrainGetModel(\'%s\')\"" %(rusinol.resultPath)], shell=True)
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
# Image rescale implementation .... method 2
# the function does the processing of each test image and saves the response
# to a pickle file
# Run on parallel cores
def parallelImageProcessing(patchSizeTraining, patchStepTraining, \
                             scaleList, testPredictFolder, svmWeights, feStandardized, enableMSER, \
                             imgPath): 
    start = time.time()
    dirName, fName = os.path.split(imgPath)  
    fName = fName.split('.')[0]
    dirName = dirName.split('/')[-1]
    print 'Starting processing Image - ', imgPath
    imageBatchSize = 500
    minT  = 0
    # save file names
    predictFileName = os.path.join(testPredictFolder, 'MaxPredict_' + dirName + '_' + fName + '.npy')
    meanFeatureValues =  feStandardized[0]
    stdDevFeatureValues =  feStandardized[1]
    if(os.path.isfile(predictFileName)):
        print 'Data already present'  
        imageArray = np.round(imread(imgPath, flatten=True)).astype(np.uint8)
        maxPredictImage = np.load(predictFileName)            
        maxPredictImage[maxPredictImage < 1.0] = 0
        plt.imshow(imageArray)
        plt.imshow(maxPredictImage, cmap = cm.get_cmap('Greys_r'), alpha = 0.8)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.show()
    else:
        # open the image and load the image as array
        imageArray = np.round(imread(imgPath, flatten=True)).astype(np.uint8)
        orgImageSize = imageArray.shape
        if (enableMSER == True):
            delta = 5
            minArea = 30
            maxArea = 90000
            maxVariation = 0.2
            minDiversity = 0.1
            # 1 column - width 2nd column height
            bboxesDetected = mserHelper.detectMSERBboxes(imgPath, delta, minArea, 
                                                         maxArea, maxVariation, minDiversity)
            numOfScales = bboxesDetected.shape[0]
            print 'Number of scales detected by MSER  in image - ', fName, ' - ', numOfScales
            if(numOfScales == 0):
                bboxesDetected = [patchSizeTraining[0],patchSizeTraining[1]]
        else:
            numOfScales = len(scaleList)
        maxPredictImage = np.ones(imageArray.shape)*(-np.Inf)
        for scaleRun in xrange(numOfScales):
            if(enableMSER == True):
                # 1 column - width 2nd column height
                curBBoxes = bboxesDetected[scaleRun,:]
                widthRatio = float(patchSizeTraining[1])/curBBoxes[0]
                heightRatio = float(patchSizeTraining[0])/curBBoxes[1]
                rescaledImgSize = (int(orgImageSize[0]*heightRatio), int(orgImageSize[1]*widthRatio))
                rescaledImage = imresize(imageArray, rescaledImgSize, interp = 'bicubic')
                print 'Computing the response for image ', fName, ' size ', rescaledImage.shape
            else:
                curScale = scaleList[scaleRun]
                #curScale = 0.5
                print 'Computing the response for image ', fName, ' at scale ', curScale
                # rescale the image based on ratio given
                rescaledImage = imresize(imageArray, curScale, interp = 'bicubic')
                widthRatio = curScale
                heightRatio = curScale
            if((rescaledImage.shape[0] >= 32) and (rescaledImage.shape[1] >=32)):
                txtPredictions = np.zeros((rescaledImage.shape[0]-24,rescaledImage.shape[1]-24))
                for rowImageRun in np.arange(0, rescaledImage.shape[0]-24, imageBatchSize-24):
                    for colImageRun in np.arange(0, rescaledImage.shape[1]-24, imageBatchSize-24):
                        curImage = rescaledImage[rowImageRun:rowImageRun+imageBatchSize, colImageRun:colImageRun+imageBatchSize]
                        frames, siftArray = vlfeat.vl_dsift(curImage.astype(np.float32), size = 8, step = 1 ,verbose = False)
                        siftArray = siftArray.T.astype(np.float32)
                        siftArray -= meanFeatureValues
                        siftArray /= stdDevFeatureValues
                        frames = frames.T
                        minColumn = np.min(frames[:,0])
                        maxColumn = np.max(frames[:,0])
                        minRow = np.min(frames[:,1])
                        maxRow = np.max(frames[:,1])
                        # +1 is beacause of zero indexing
                        numColumns = maxColumn - minColumn + 1
                        numRows  = maxRow -minRow + 1
                        siftArray = np.hstack((siftArray, np.ones((siftArray.shape[0],1))))
                        predict = np.dot(siftArray, svmWeights)
                        predictPart = predict[:,0].reshape(numColumns,numRows).T
                        txtPredictions[rowImageRun:rowImageRun+numRows,\
                                        colImageRun:colImageRun+numColumns] = predictPart
#                 tempCopy = txtPredictions.copy()
#                 tempCopy[tempCopy < 0 ] = 0
#                 plt.imshow(tempCopy, cmap = cm.get_cmap('Greys_r'))
#                 plt.show()
#                 eachCountNum = 0
#                 fileRun = 0 
                arrIndex = np.ndindex(txtPredictions.shape[0], txtPredictions.shape[1])
                for predictRun in arrIndex:
                    eachPredict = txtPredictions[predictRun[0], predictRun[1]]
                    minT = np.minimum(minT, eachPredict)
                    eachRowCord = np.floor((predictRun[0]*patchStepTraining[0])/heightRatio)
                    eachColCord = np.floor((predictRun[1]*patchStepTraining[1])/widthRatio)
                    eachRowCordEnd = eachRowCord + np.floor(patchSizeTraining[0]/heightRatio)
                    eachColCordEnd = eachColCord + np.floor(patchSizeTraining[1]/widthRatio)
                    predPart = maxPredictImage[eachRowCord:eachRowCordEnd, eachColCord:eachColCordEnd]
                    maxPredictImage[eachRowCord:eachRowCordEnd, eachColCord:eachColCordEnd] = np.maximum(predPart, eachPredict)
                    # for video creation
#                     if(eachCountNum%1.0 == 0):
# #                             imageArrayTemp = imageArray.copy()
# #                             cv.rectangle(imageArrayTemp, (eachColCord,eachRowCord),(eachColCordEnd,eachRowCordEnd), (0, 0, 0), 2)
# #                             plt.clf()
# #                             plt.imshow(imageArrayTemp)
#                         imageArrayTemp = rescaledImage.copy()
#                         cv.rectangle(imageArrayTemp, (predictRun[1],predictRun[0]),(predictRun[1]+32,predictRun[0]+32), (0, 0, 0), 2)
#                         plt.clf()
#                         plt.imshow(imageArrayTemp)
#                         if(eachPredict > 0):
#                             # add text box... saying text
#                             plt.text(150, 150, 'TEXT',horizontalalignment='center', fontsize=30, weight = 'heavy', backgroundcolor='g', color='k' )
#                         else:
#                             # add text box... saying non text
#                             plt.text(150, 150, 'NON-TEXT',horizontalalignment='center', fontsize=30, weight = 'heavy', backgroundcolor='r', color='k' )
#                         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#                         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#                         plt.savefig('temp/img_'+ str(fileRun).zfill(4)+'.png')
#                         fileRun += 1
#                     eachCountNum +=1
        # save the response
        maxPredictImage[maxPredictImage==-np.Inf] = minT
        np.save(predictFileName, maxPredictImage)
        maxPredictImage[maxPredictImage < 0 ] = 0
        plt.imshow(imageArray)
        plt.imshow(maxPredictImage, cmap = cm.get_cmap('Greys_r'), alpha = 0.8)
        plt.show()
    end = time.time()
    print  'Finished processing  - ', imgPath, ' seconds - ', end-start
    
# test folder
testPredictFolder = os.path.join(rusinol.resultPath, 'prediction')
if not os.path.exists(testPredictFolder):
    os.makedirs(testPredictFolder)
# xml file location
xmlName = 'segmentation.xml'
testXmlCompletePath = os.path.join(rusinol.testPath, xmlName)
# scales 
scaleList = [0.1, 0.2 , 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
# now we have trained our classifier 
imagePathSaveList = os.path.join(rusinol.testPath, 'imgPathList.pkl')
with open(imagePathSaveList, 'rb') as pklFile:
    imgPathList = pickle.load(pklFile)
    pklFile.close()
imgPathList = imgPathList[:10]
imgPathList[0] = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','test','ryoungt_05.08.2002','PICT0015.JPG')
#imgPathList[0] = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','test','ryoungt_13.08.2002','dPICT0062.JPG')
numOfImages = len(imgPathList)
# parallel processing
numCores = joblib.cpu_count()
if numCores == 8:
    numJobs = 5
elif numCores == 4:   
    numJobs = 2
else:
    numJobs = numCores/2    
print 'Running jobs in parallel on cores - ', numJobs

# parallel job scheduler    
joblib.Parallel(n_jobs = numJobs)(joblib.delayed(parallelImageProcessing)(rusinol.cellSize, \
                        rusinol.stepSize, scaleList, testPredictFolder, \
                        pixelTextProbPredictorWeights, feStandardized, scalesFromMSER, eachImgPath) for eachImgPath in imgPathList)
# exit the script
exit()
