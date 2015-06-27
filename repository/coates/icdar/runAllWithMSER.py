import cPickle as pickle
import numpy as np
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import Image
import coates
import helper
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.cross_validation import StratifiedKFold
import xml.etree.ElementTree as xmlParser
from scipy.misc import imresize
import multiprocessing

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
trainingParamsPath = os.path.join(coates.resultPath,'trainingParams.pkl')
tData_idfier = str(coates.cellSize[0]) + '_' + str(coates.cellSize[1])
dictData_idfier = str(coates.subCellSize[0]) + '_' + str(coates.subCellSize[1])
# dictionary data points save location
dictDataPointsPath = os.path.join(coates.trainPath, 'dictDataPoints_' + dictData_idfier +'.npy')  
# positive and negative patch save location
positivePatchesSavePath = os.path.join(coates.trainPath, 'positivePatches_'+ tData_idfier + '.npy')
negativePatchesSavePath = os.path.join(coates.trainPath, 'negativePatches_'+ tData_idfier + '.npy')
positivePatchesTestSavePath = os.path.join(coates.trainPath, 'positivePatchesTest_'+ tData_idfier + '.npy')
negativePatchesTestSavePath = os.path.join(coates.trainPath, 'negativePatchesTest_'+ tData_idfier + '.npy')

if(os.path.isfile(dictionaryPath)):
    print 'Dictionary already generated. Hence Loading it from ', dictionaryPath
    finalDict = np.load(dictionaryPath) 
    print 'Shape of the dictionary - ', finalDict.shape  
    # load the training params obtained during training
    print 'Loading Training parameters from ', trainingParamsPath
    with open(trainingParamsPath, 'rb') as pklFile:
       trainingParamsList = pickle.load(pklFile)
    pklFile.close()
    print 'Number of params in the List - ', len(trainingParamsList)
    
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
    
    plt.clf()
    eigValues = abs(np.linalg.eigh(np.cov(dictDataPoints.T))[0])
    plt.scatter(np.arange(eigValues.shape[0]),eigValues)
    plt.savefig(os.path.join(coates.resultPath, 'eigenValuesRawData.png'))
                  
    # prepare the inputs for training - coates.constantZCA
    dictDataPoints, trainingParamsList =  coates.prepareData(dictDataPoints, coates.constantZCA)
    
    print 'Covariance matrix of whitened data'
    print  np.diag(np.cov(dictDataPoints.T))
    # save the figure of whitened data points
    imgs = np.reshape(dictDataPoints[randomSelect],(numSelect, coates.subCellSize[0], \
                                                        coates.subCellSize[1]))
    helper.displayImgs(imgs,(8, 8))  
    plt.savefig(os.path.join(coates.resultPath, 'whitenedDataPoints.png'))
   
    # save the eigen value graph
    plt.clf()
    eigValues = abs(np.linalg.eigh(np.cov(dictDataPoints.T))[0])
    plt.scatter(np.arange(eigValues.shape[0]),eigValues)
    plt.savefig(os.path.join(coates.resultPath, 'eigenValuesWhitenedData.png'))
       
    print 'Number of params in the List - ', len(trainingParamsList)
    print 'Saving Training parameters'
    
    # save the zca params list
    with open(trainingParamsPath, 'wb') as pklFile:
        pickle.dump(trainingParamsList, pklFile)
    pklFile.close()            
    
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
feScalerPath = os.path.join(coates.resultPath, 'feScaler.pkl')
if(os.path.isfile(trainDataPath)):
    print 'Training Data already generated.Loading from ', trainDataPath
    trainDataSet = np.load(trainDataPath)
    print 'Shape of the Training data - ', trainDataSet.shape
    print 'Test Data already generated.Loading from ', testDataPath
    testDataSet = np.load(testDataPath)
    print 'Shape of the Test data - ', testDataSet.shape
    featureScaler =  joblib.load(feScalerPath) 
else:
    print 'Generating Training Data.'
    print 'Loading - ',positivePatchesSavePath
    print 'Loading - ',negativePatchesSavePath
    positivePatches = np.load(positivePatchesSavePath)
    negativePatches = np.load(negativePatchesSavePath)
    print 'Text Patches - ', positivePatches.shape
    print 'Non Text Patches - ', negativePatches.shape
    # Now we have generated the required number of text patches and non text patches
    # Generating features
    # consists of preprocessing ,ZCA whitening, Average pooling operations
    # retuns 2d array ....argument is 3d array
#     positivePatches = coates.generateFeatures(positivePatches, finalDict, coates.subCellSize, \
#                                               coates.subStepSize, trainingParamsList)

    positivePatches = coates.computeResponsePatches(positivePatches, finalDict, coates.cellSize,\
                                           coates.stepSize, coates.subCellSize, coates.subStepSize, \
                                           trainingParamsList)
    
    # Generating features
    # consists of preprocessing ,ZCA whitening, Average pooling operations
    # retuns 2d array ....argument is 3d array
#     negativePatches = coates.generateFeatures(negativePatches, finalDict, coates.subCellSize, \
#                                               coates.subStepSize, trainingParamsList)
    
    negativePatches = coates.computeResponsePatches(negativePatches, finalDict, coates.cellSize,\
                                           coates.stepSize, coates.subCellSize, coates.subStepSize, \
                                           trainingParamsList)
    
    # stack up ...to build the training data
    # training data shape - Number of samples X features dimension
    positivePatches = np.hstack((np.ones((positivePatches.shape[0],1)), positivePatches))
    negativePatches = np.hstack((np.zeros((negativePatches.shape[0],1)), negativePatches))
    # save the taining set
    # training data shape - Number of samples X features dimension
    trainDataSet = np.vstack((positivePatches, negativePatches))
    outputlabels = trainDataSet[:, 0]
    outputlabels = outputlabels[:,np.newaxis]
    # features dimension X Number of samples 
    #trainDataSet, svmParams = coates.standardizeFeatures(trainDataSet[:,1:].T)
    featureScaler = StandardScaler()
    trainDataSet = featureScaler.fit_transform(trainDataSet[:,1:])
    trainDataSet = np.hstack((outputlabels, trainDataSet))
    print 'Mean (training data) - ', np.mean(trainDataSet[:,1:])
    print 'Std Dev (training data)- ', np.std(trainDataSet[:,1:])
    # save the model for prediction
    joblib.dump(featureScaler, feScalerPath) 
        
    if(np.isnan(trainDataSet).sum()):
        print 'Warning - NaNs in the training data'    
    print 'Saving training data'
    print 'Positive training examples - ',positivePatches.shape
    print 'Negative training examples - ',negativePatches.shape
    print 'Training Data Shape - ', trainDataSet.shape
    np.save(trainDataPath, trainDataSet)  
    # for test
    # test
    print 'Generating Testing Data.'
    print 'Loading - ',positivePatchesTestSavePath
    print 'Loading - ',negativePatchesTestSavePath
    positivePatchesTest = np.load(positivePatchesTestSavePath)
    negativePatchesTest = np.load(negativePatchesTestSavePath)
    print 'Text Patches - ', positivePatchesTest.shape
    print 'Non Text Patches - ', negativePatchesTest.shape
    
#     positivePatchesTest = coates.generateFeatures(positivePatchesTest, finalDict, \
#                                                   coates.subCellSize, coates.subStepSize, \
#                                                   trainingParamsList)

    positivePatchesTest = coates.computeResponsePatches(positivePatchesTest, finalDict, coates.cellSize,\
                                           coates.stepSize, coates.subCellSize, coates.subStepSize, \
                                           trainingParamsList)

    # Generating features
    # consists of preprocessing ,ZCA whitening, Average pooling operations
    # retuns 2d array ....argument is 3d array
#     negativePatchesTest = coates.generateFeatures(negativePatchesTest, finalDict,
#                                               coates.subCellSize, coates.subStepSize, \
#                                               trainingParamsList)

    negativePatchesTest = coates.computeResponsePatches(negativePatchesTest, finalDict, coates.cellSize,\
                                           coates.stepSize, coates.subCellSize, coates.subStepSize, \
                                           trainingParamsList)   
    
    positivePatchesTest = featureScaler.transform(positivePatchesTest)
    negativePatchesTest = featureScaler.transform(negativePatchesTest)
        
    positivePatchesTest = np.hstack((np.ones((positivePatchesTest.shape[0],1)), positivePatchesTest))
    negativePatchesTest = np.hstack((np.zeros((negativePatchesTest.shape[0],1)), negativePatchesTest))
    
    # save the taining set
    # training data shape - Number of samples X features dimension
    testDataSet = np.vstack((positivePatchesTest, negativePatchesTest))
    print 'Mean (test data) - ', np.mean(testDataSet[:,1:])
    print 'Std Dev (test data)- ', np.std(testDataSet[:,1:]) 
    print 'Test Data Shape - ', testDataSet.shape
    np.save(testDataPath, testDataSet) 
    # memory clean up
    del negativePatchesTest
    del positivePatchesTest   
    del negativePatches 
    del positivePatches    
print '############################################################################################'
###################################################################################################
# training the SVM
svmModelPath = os.path.join(coates.resultPath, 'SVMModelSave.pkl')
if(os.path.isfile(svmModelPath)):
    print 'SVM Model already generated'
    print 'Loading the SVM Model - ', svmModelPath
    pixelTextProbPredictor = joblib.load(svmModelPath) 
    print 'Selected Model C value - ', pixelTextProbPredictor.C 
    print 'Model Score on test data(unseen) - ', pixelTextProbPredictor.score(testDataSet[:,1:], \
                                                                              testDataSet[:,0])
else:
    # grid search
    print 'Training the SVM'
    # shuffle
    dataIndex = np.arange(trainDataSet.shape[0])
    np.random.shuffle(dataIndex)
    np.random.shuffle(dataIndex)
    trainDataSet = trainDataSet[dataIndex,:]
    dataIndex = np.arange(testDataSet.shape[0])
    np.random.shuffle(dataIndex)
    np.random.shuffle(dataIndex)
    testDataSet = testDataSet[dataIndex,:]
    CVals = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    errorCollect = []
    modelCollect = []
    precisionCollect = []
    for curCval in CVals:
        pixelTextProbPredictor = LinearSVC(tol=0.00001)
        #pixelTextProbPredictor = SVC(kernel='linear', probability=True, max_iter=1000, tol=0.00001)
        pixelTextProbPredictor.verbose = 1
        pixelTextProbPredictor.C = curCval
        #11 training data shape - Number of samples X features dimension
        pixelTextProbPredictor = pixelTextProbPredictor.fit(trainDataSet[:,1:], trainDataSet[:,0])
        # testing data
        score = 0
        error = 0
        for testRun in np.arange(3):
            randomSelect = np.random.randint(0, testDataSet.shape[0], 8000)
            score += pixelTextProbPredictor.score(testDataSet[randomSelect,1:], testDataSet[randomSelect,0])
            predictValues = pixelTextProbPredictor.predict(testDataSet[randomSelect, 1:])
            error += np.mean(predictValues != testDataSet[randomSelect,0])
        precisionCollect.append(score/3.0)
        errorCollect.append(error/3.0)
        modelCollect.append(pixelTextProbPredictor)
            
    selectIndex = precisionCollect.index(max(precisionCollect))
    pixelTextProbPredictor =  modelCollect[selectIndex] 
    print 'Selected C value - ', CVals[selectIndex]
    print 'Model Precision - ', precisionCollect[selectIndex]
    print 'Model Error - ', errorCollect[selectIndex] 
    # save the model for prediction
    joblib.dump(pixelTextProbPredictor, svmModelPath) 
    # memory clean up 
    del modelCollect
    del precisionCollect
    del errorCollect   

# using grid seach  sk learn .....eats up all the  memory for large data sets
#   parameters = {'C' : [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]}
#   parameters = {'C' : np.logspace(-4, 2, 7)}
#   bestModelEstimator = grid_search.GridSearchCV(SVC(kernel='linear', probability=True, max_iter = 1000),\
#                                               parameters, n_jobs=-1, verbose=100, cv = 2)
#   bestModelEstimator = grid_search.GridSearchCV(LinearSVC(), \
#                                                   parameters, n_jobs=1, verbose=100, \
#                                                   cv = StratifiedKFold(y=trainDataSet[:,0], n_folds=3)) 
#   bestModelEstimator.fit(trainDataSet[:,1:],trainDataSet[:,0])
#   pixelTextProbPredictor = bestModelEstimator.best_estimator_ 
#   print 'Selected Model C value - ', pixelTextProbPredictor.C 
#   print 'Model Score on test data(unseen) - ', pixelTextProbPredictor.score(testDataSet[:,1:], \
#                                                                               testDataSet[:,0])
del testDataSet
del trainDataSet
#time.sleep(60)
print '############################################################################################'

def parallelImageProcessing(): 




# wait for one minute ............... image rescale implementation .... method 2
testPredictFolder = os.path.join(coates.resultPath, 'prediction')
if not os.path.exists(testPredictFolder):
    os.makedirs(testPredictFolder)
# xml file location
xmlName = 'locations.xml'
testXmlCompletePath = os.path.join(coates.testPath, xmlName)
scaleList = [0.1, 0.2 , 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
#scaleList = [0.5, 1.0, 1.5]
minClassifierValue = 0
maxClassifierValue = 0
# now we have trained our classifier 
print 'Reading Locations Xml from ', testXmlCompletePath
tree = xmlParser.parse(testXmlCompletePath)
root = tree.getroot()
numOfImages = root.__len__()
imageRun = 1
for imageDetails in list(root):
    print '************************************************************************'
    start = time.time()
    # get the image name
    imgName = imageDetails.find('imageName').text
    imgPath = os.path.join(coates.testPath, imgName)
    dirName, fName = os.path.split(imgPath)  
    fName = fName.split('.')[0]
    dirName = dirName.split('/')[-1]
    print imageRun, '. Image - ', imgPath , '(' , imageRun, '/', numOfImages,')'
    # save file names
    predictFileName = os.path.join(testPredictFolder, 'Predict_' + dirName + '_' + fName + '.npz')
         
    if(os.path.isfile(predictFileName)):
        print 'Data already present'  
    else:
        # open the image and load the image as array
        image = Image.open(imgPath).convert('L')
        imageArray = np.asarray(image, dtype=np.uint8) 
        rowOrg= imageArray.shape[0]
        colOrg = imageArray.shape[1]
        # huge array
        # create an array for storage
        predictImageBinMap = np.zeros((rowOrg, colOrg, len(scaleList)))        
        for scaleRun in np.arange(len(scaleList)):
            curScale = scaleList[scaleRun]
            print 'Computing the response for image ', fName, \
                  '(' , imageRun, '/', numOfImages,')', ' at scale ', curScale
            # rescale the image based on ratio given
            rescaledImage = imresize(imageArray.astype(np.float64), curScale, interp = 'bicubic')
            # zero padding
            rescaledImage = helper.padImage(rescaledImage, imageArray.shape, curScale)
            # features
            features = coates.computeResponseImage(rescaledImage, finalDict, coates.cellSize, coates.stepSize, \
                                   coates.subCellSize, coates.subStepSize, trainingParamsList)           
#             # sample patches of required scale
#             patchCollectArray = coates.denseSampler(rescaledImage, patchSizeTraining, coates.stepSize)
#             # feature generation
#             # consists of preprocessing ,ZCA whitening, Average pooling operations
#             # input 4d matrix ...
#             features = coates.generateFeatures(patchCollectArray, finalDict, coates.subCellSize, \
#                                                coates.subStepSize, trainingParamsList)  
          
            # declare an array of required size
            rowCord, colCord = np.ogrid[0:(rowOrg-coates.cellSize[0]+1):coates.stepSize[0], \
                                        0:(colOrg-coates.cellSize[1]+1):coates.stepSize[1]]            
            predictEachScale = np.zeros((rowOrg, colOrg))                                                 
            arrIndex = np.ndindex(features.shape[0], features.shape[1]) 
            for featureRun in arrIndex:
                eachFeature = featureScaler.transform(features[featureRun])                
                predict = pixelTextProbPredictor.decision_function(eachFeature)
                #predict = pixelTextProbPredictor.predict_proba(eachFeature)[0,1]
                if(np.isnan(predict)):
                    print 'Warning - NaN predicted by SVM'                    
                # prediction from Linear classifier 
                predictEachScale[rowCord[featureRun[0], 0]:rowCord[featureRun[0], 0]+ \
                                   coates.cellSize[0], \
                                   colCord[0, featureRun[1]]:colCord[0, featureRun[1]]+ \
                                   coates.cellSize[1]] += predict
            # pad to single shape                            
            predictImageBinMap[:,:,scaleRun] = predictEachScale
            minClassifierValue = np.minimum(minClassifierValue,np.min(predictEachScale))
            maxClassifierValue = np.maximum(maxClassifierValue,np.max(predictEachScale))
        # save the prediction and bounding boxes
        np.savez_compressed(predictFileName, predictImageBinMap = predictImageBinMap)
        del predictImageBinMap
    end = time.time()
    print imageRun, '. Elapsed Time - ', end-start 
    imageRun = imageRun +1
print 'Max Classifier Value - ', maxClassifierValue
print 'Min Classifier Value - ', minClassifierValue

   
###################################################################################################
# wait for one minute ............... no rescale implementation .... method 1
# testPredictFolder = os.path.join(coates.resultPath, 'prediction')
# if not os.path.exists(testPredictFolder):
#     os.makedirs(testPredictFolder)
# # xml file location
# xmlName = 'locations.xml'
# testXmlCompletePath = os.path.join(coates.testPath, xmlName)
# scaleList = [(16,16),(32,32),(64,64)]
# # now we have trained our classifier 
# print 'Reading Locations Xml from ', testXmlCompletePath
# tree = xmlParser.parse(testXmlCompletePath)
# root = tree.getroot()
# numOfImages = root.__len__()
# imageRun = 1
# for imageDetails in list(root):
#     print '************************************************************************'
#     start = time.time()
#     # get the image name
#     imgName = imageDetails.find('imageName').text
#     imgPath = os.path.join(coates.testPath, imgName)
#     dirName, fName = os.path.split(imgPath)  
#     fName = fName.split('.')[0]
#     dirName = dirName.split('/')[-1]
#     print imageRun, '. Image - ', imgPath , '(' , imageRun, '/', numOfImages,')'
#     # save file names
#     predictFileName = os.path.join(testPredictFolder, 'Predict_' + dirName + '_' + fName)
#            
#     if(os.path.isfile(predictFileName)):
#         print 'Data already present'  
#     else:
#         # open the image and load the image as array
#         image = Image.open(imgPath).convert('L')
#         imageArray = np.asarray(image, dtype=np.uint8)    
#         rowMax= imageArray.shape[0]
#         colMax = imageArray.shape[1]
#         # create an array for storage
#         predictImageBinMap = np.zeros((imageArray.shape[0], imageArray.shape[1], len(scaleList))) 
#         bboxesList = [] 
#         for scaleRun in np.arange(len(scaleList)):
#             curScale = scaleList[scaleRun]
#             # sample patches of required scale
#             patchCollectArray = coates.denseSampler(imageArray, curScale, curScale)        
#                        
#             # now generate features
#             # feature generation
#             # consists of preprocessing ,ZCA whitening, Average pooling operations
#             # input 4d matrix ...
#             features = coates.generateFeatures(patchCollectArray, finalDict, coates.subCellSize, \
#                                                coates.subStepSize, trainingParamsList)            
#             # declare an array of required size
#             rowCord, colCord = np.ogrid[0:(rowMax-curScale[0]+1):curScale[0], \
#                                         0:(colMax-curScale[1]+1):curScale[1]]
#               
#             arrIndex = np.ndindex(features.shape[0], features.shape[1])    
#             for featureRun in arrIndex:
#                 eachFeature = featureScaler.transform(features[featureRun])                
#                 #predict = pixelTextProbPredictor.predict_proba(eachFeature)[0,1]                
#                 predict = pixelTextProbPredictor.decision_function(eachFeature)
#                 if(np.isnan(predict)):
#                     print 'Warning - NaN predicted by SVM'                    
#                 # prediction from Linear classifier 
#                 predictImageBinMap[rowCord[featureRun[0],0]:rowCord[featureRun[0],0]+curScale[0], \
#                                    colCord[0,featureRun[1]]:colCord[0,featureRun[1]]+curScale[1], \
#                                    scaleRun] = predict
#                 if predict > 0 :                  
#                     colX = colCord[0,featureRun[1]]
#                     rowY = rowCord[featureRun[0],0]
#                     widthX = curScale[1]
#                     heightY = curScale[0]
#                     rect = [colX, rowY, widthX, heightY, predict]
#                     bboxesList.append(rect)                   
#              
#         # save the prediction and bounding boxes
#         bboxesList = np.array(bboxesList)
#         np.savez_compressed(predictFileName, predictImageBinMap = predictImageBinMap, \
#                              bboxesList = bboxesList)
#         del predictImageBinMap
#         del bboxesList            
#     end = time.time()
#     print imageRun, '. Elapsed Time - ', end-start 
#     imageRun = imageRun +1
###################################################################################################
# Method 3 ..... patch rescale method
# # start the test process ... patch rescaled implementation of the patch -  method 3
# testPredictFolder = os.path.join(coates.resultPath, 'prediction')
# if not os.path.exists(testPredictFolder):
#     os.makedirs(testPredictFolder)
# # xml file location
# xmlName = 'locations.xml'
# testXmlCompletePath = os.path.join(coates.testPath, xmlName)
# scaleList = [(16,16), (32,32), (64,64)]
# # no scales bigger that the image
# #scaleList = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
# # now we have trained our classifier 
# print 'Reading Locations Xml from ', testXmlCompletePath
# tree = xmlParser.parse(testXmlCompletePath)
# root = tree.getroot()
# numOfImages = root.__len__()
# imageRun = 1
# for imageDetails in list(root):
#     print '****************************************************************************************'
#     start = time.time()
#     # get the image name
#     imgName = imageDetails.find('imageName').text
#     imgPath = os.path.join(coates.testPath, imgName)
#     dirName, fName = os.path.split(imgPath)  
#     fName = fName.split('.')[0]
#     dirName = dirName.split('/')[-1]
#     print imageRun, '. Image - ', imgPath , '(' , imageRun, '/', numOfImages,')'
#     # save file names
#     predictFileName = os.path.join(testPredictFolder, 'Predict_' + dirName + '_' + fName + '.npy')             
#     if(os.path.isfile(predictFileName)):
#         print 'Data already present'  
#     else:
#         # open the image and load the image as array
#         image = Image.open(imgPath).convert('L')
#         orgImageArray = np.asarray(image, dtype=np.uint8)
#         orgImSize = orgImageArray.shape
#         # create an array for storage
#         predictImageBinMap = np.zeros((orgImageArray.shape[0], \
#                                        orgImageArray.shape[1], len(scaleList)))
#         # collected the patches
#         patchSizeTraining = coates.cellSize 
#         patchCollectArray = coates.denseSampler(orgImageArray, patchSizeTraining, patchSizeTraining)
#         # declare an array of required size
#         rowCord, colCord = np.ogrid[0:(orgImSize[0]-patchSizeTraining[0]+1):patchSizeTraining[0], \
#                                             0:(orgImSize[1]-patchSizeTraining[1]+1):patchSizeTraining[1]]        
#         # run through the patches
#         arrIndex = np.ndindex(patchCollectArray.shape[0], patchCollectArray.shape[1])
#         for patchRun in arrIndex:
#             curPatch = patchCollectArray[patchRun]
#             # run through the scale list
#             scaleRun = 0
#             for eachScale in scaleList:
#                 # same matlab resize
#                 patchPil = Image.fromarray(curPatch)
#                 reScaledPatch =  np.asarray(patchPil.resize(eachScale, resample=Image.BICUBIC))
#                 reScaledPatch = reScaledPatch[np.newaxis,:,:]
#                 # generate features
#                 features = coates.generateFeatures(reScaledPatch, finalDict, coates.subCellSize, \
#                                                            coates.subStepSize, coates.numOfClusters, \
#                                                            trainingParamsList)            
#                 eachFeature = featureScaler.transform(features)                
#                 #predict = pixelTextProbPredictor.predict_proba(eachFeature)[0,1]                
#                 predict = pixelTextProbPredictor.decision_function(eachFeature)
#                 if(np.isnan(predict)):
#                     print 'Warning - NaN predicted by SVM'                    
#                 # prediction from Linear classifier 
#                 predictImageBinMap[rowCord[patchRun[0],0]:rowCord[patchRun[0],0]+patchSizeTraining[0], \
#                                    colCord[0,patchRun[1]]:colCord[0,patchRun[1]]+patchSizeTraining[1], \
#                                    scaleRun] = predict 
#                 scaleRun += 1                              
#         # save the prediction
#         np.save(predictFileName, predictImageBinMap)
#         del predictImageBinMap
#     # time stats                
#     end = time.time()
#     print imageRun, '. Elapsed Time - ', end-start 
#     imageRun += 1 
################################################################################################### 
# exit the script
exit()   
    
    