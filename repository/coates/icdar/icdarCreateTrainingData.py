import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import time
from scipy.misc import imread
import cPickle as pickle
import coates
import helper

# identifier for training data and dictionary data
tData_idfier = str(coates.cellSize[0]) + '_' + str(coates.cellSize[1])  
dictData_idfier = str(coates.subCellSize[0]) + '_' + str(coates.subCellSize[1])
# dictionary data points save location
dictDataPointsPath = os.path.join(coates.trainPath, 'dictDataPoints_' + dictData_idfier +'.npy')
# positive and negative patch save location
positivePatchesSavePath = os.path.join(coates.trainPath, 'positivePatches_'+ tData_idfier + '.npy')
negativePatchesSavePath = os.path.join(coates.trainPath, 'negativePatches_'+ tData_idfier + '.npy')
# test
positivePatchesTestSavePath = os.path.join(coates.trainPath, 'positivePatchesTest_'+ tData_idfier + '.npy')
negativePatchesTestSavePath = os.path.join(coates.trainPath, 'negativePatchesTest_'+ tData_idfier + '.npy')
# check if all the data has been generated
checkPresence = os.path.isfile(dictDataPointsPath) and \
                os.path.isfile(positivePatchesSavePath) and \
                os.path.isfile(negativePatchesSavePath) and \
                os.path.isfile(positivePatchesTestSavePath) and \
                os.path.isfile(negativePatchesTestSavePath)
imagePathSaveList = os.path.join(coates.trainPath, 'imgPathList.pkl')
charBboxInfosFolder = os.path.join(coates.trainPath,'charBboxInfos')
if(checkPresence):
    print 'Training Data already present'
else:
    # parsing the XML file
    print 'Generating the Training Data'
    with open(imagePathSaveList, 'rb') as pklFile:
        imgPathList = pickle.load(pklFile)
        pklFile.close()
    numOfImages = len(imgPathList)
    # calculate the number postive and negative patches 
    numPositivePatchFromImage = (coates.numPositiveTextPatches/numOfImages) + 1 
    numNegativePatchFromImage = (coates.numNegativeTextPatches/numOfImages) + 1
    imageRun = 1
    positivePatches = []
    negativePatches = []
    # loop through each image 
    ###################################################################################################
    for imagePath in imgPathList:
        print '************************************************************************************'
        start = time.time()
        dirName, fName = os.path.split(imagePath)
        fName = fName.split('.')[0]
        dirName = dirName.split('/')[-1]
        print imageRun, '. Image - ', imagePath , '(' , imageRun, '/', numOfImages,')'
        # open the corresponding file wher char info i saved
        charInfoFileName = os.path.join(charBboxInfosFolder, 'CharInfos_' + dirName + '_' + fName + '.npy')
        charInfos = np.load(charInfoFileName)
        # open the image
        imageArray = np.round(imread(imagePath, flatten=True)).astype(np.uint8)
        trainPatchStep = (1,1)
        patchCollectArray = coates.denseSampler(imageArray, coates.cellSize, trainPatchStep)
        # update the text presense based on ground truth
        textPresenceArray = coates.updateTextPresence(charInfos, imageArray.shape, \
                                                      coates.cellSize, trainPatchStep) 
        # find non text patches and save for later use ...training data
        nonTextIndexArray = np.asarray(np.where(textPresenceArray == 0),dtype = np.int32).T
        # get the data form training
        temp = np.arange(nonTextIndexArray.shape[0])
        np.random.shuffle(temp)
        nontextIndexSelectTdata = temp[0:numNegativePatchFromImage]
        #nontextIndexSelectTdata = np.random.random_integers(0, nonTextIndexArray.shape[0]-1, numNegativePatchFromImage)
        auxVar = nonTextIndexArray[nontextIndexSelectTdata]
        nonTextSelectPatches = patchCollectArray[auxVar[:,0],auxVar[:,1],:,:]
        negativePatches.extend(nonTextSelectPatches)
        # text patches
        # selecting the necessary patches from the list
        textIndexArray = np.asarray(np.where(textPresenceArray == 1), dtype = np.int32).T
        if(textIndexArray.shape[0] == 0):
            print 'No Text Patches in the Image.'
        else:
            # select randomly  
            #textIndexSelectTdata = np.random.random_integers(0, textIndexArray.shape[0]-1, numPositivePatchFromImage)
            # text patches for training data
            auxVar = textIndexArray
            textSelectPatches = patchCollectArray[auxVar[:,0],auxVar[:,1],:,:]
#             for temprun in xrange(textSelectPatches.shape[0]):
#                 plt.imshow(textSelectPatches[temprun,:,:])
#                 plt.show()
            positivePatches.extend(textSelectPatches)    
        end = time.time()
        print imageRun, '. Time elapsed - ', end-start
        imageRun = imageRun + 1 
    print '############################################################################################'
    # saving data
    # saving the positive and negative text patches
    numTextPatches = 30000
    numNonTextPatches = 60000
    numTestPatches = 60000
    numSelect = 256  # for figures
    layOut = (16, 16)
    positivePatches = np.array(positivePatches, dtype = np.uint8) 
    negativePatches = np.array(negativePatches, dtype = np.uint8)
    print 'Harvested Positive patches from dataset  - ', positivePatches.shape[0]
    print 'Harvested Negative patches from dataset  - ', negativePatches.shape[0]
    # text and non text pactches
    selectIndices = np.arange(positivePatches.shape[0])
    np.random.shuffle(selectIndices)
    np.random.shuffle(selectIndices)
    np.random.shuffle(selectIndices)
    #selectIndices = np.random.random_integers(0, positivePatches.shape[0]-1, positivePatches.shape[0])
    positivePatchesSelect = positivePatches[selectIndices[:numTextPatches],:,:]
    positiveTestPatches = positivePatches[selectIndices[numTextPatches:],:,:]
    positiveTestPatches = positiveTestPatches[:numTestPatches]    
    selectIndices = np.arange(negativePatches.shape[0])
    np.random.shuffle(selectIndices)
    np.random.shuffle(selectIndices)
    np.random.shuffle(selectIndices)
    #selectIndices = np.random.random_integers(0, negativePatches.shape[0]-1, negativePatches.shape[0])
    negativePatchesSelect = negativePatches[selectIndices[:numNonTextPatches],:]
    negativeTestPatches = negativePatches[selectIndices[numNonTextPatches:],:,:]
    negativeTestPatches = negativePatches[:numTestPatches]
    
    print 'Saving training data'
    print 'Positive training examples - ', positivePatchesSelect.shape
    print 'Negative training examples - ', negativePatchesSelect.shape
    np.save(positivePatchesSavePath, positivePatchesSelect)
    np.save(negativePatchesSavePath, negativePatchesSelect)
            
    print 'Positive Test training examples - ', positiveTestPatches.shape
    print 'Negative Test training examples - ', negativeTestPatches.shape
    np.save(positivePatchesTestSavePath, positiveTestPatches)
    np.save(negativePatchesTestSavePath, negativeTestPatches)  
    
    # text patches
    randomSelect = np.random.random_integers(0, positivePatchesSelect.shape[0]-1, numSelect)
    imgs = np.reshape(positivePatchesSelect[randomSelect],(numSelect, coates.cellSize[0], coates.cellSize[1]))
    helper.displayImgs(imgs, layOut)  
    plt.savefig(os.path.join(coates.trainPath,'txtPatches_' + str(tData_idfier) +'.png'))
    
    # non text patches
    plt.clf()
    randomSelect = np.random.random_integers(0, negativePatchesSelect.shape[0]-1, numSelect)
    imgs = np.reshape(negativePatchesSelect[randomSelect],(numSelect, coates.cellSize[0], coates.cellSize[1]))
    helper.displayImgs(imgs, layOut)  
    plt.savefig(os.path.join(coates.trainPath,'nontxtPatches_' + str(tData_idfier) +'.png')) 
   
    # data points for dictionary creation 
    numOfPatchesPerImage = 8 
    dictDataPoints = np.zeros((numOfPatchesPerImage*positivePatchesSelect.shape[0], \
                               coates.subCellSize[0]*coates.subCellSize[1]), dtype = np.uint8)
    dictRun = 0
    for patchRun in xrange(positivePatchesSelect.shape[0]):
        if not np.mod(patchRun, 5000):
            print  'Extracting patches - ', patchRun, '/' , positivePatchesSelect.shape[0]
        cntPerImage  = 0
        while cntPerImage < 8:  
            rowNum = np.random.random_integers(0, positivePatchesSelect.shape[1] - coates.subCellSize[0])
            colNum = np.random.random_integers(0, positivePatchesSelect.shape[2] - coates.subCellSize[1])   
            dictDataPoints[dictRun,:] = positivePatchesSelect[patchRun, rowNum:rowNum+coates.subCellSize[0],\
                                    colNum:colNum+coates.subCellSize[1]].flatten()
            cntPerImage += 1
            dictRun += 1
    print 'Dimension - ', dictDataPoints.shape[1]
    print 'Number of Data Points - ', dictDataPoints.shape[0]  
    print 'Saving Dictionary Data points.'
    np.save(dictDataPointsPath, dictDataPoints)
    
    # dictionary figure
    plt.clf() 
    randomSelect = np.random.random_integers(0, dictDataPoints.shape[0]-1, numSelect)
    imgs = np.reshape(dictDataPoints[randomSelect],(numSelect, coates.subCellSize[0], \
                                                coates.subCellSize[1]))
    helper.displayImgs(imgs,layOut)  
    plt.savefig(os.path.join(coates.trainPath, 'dictionaryDataPoints_' + str(dictData_idfier) +'.png'))
    
    # memory clean up
    del imgs
    del dictDataPoints
    del positiveTestPatches
    del negativeTestPatches
    del positivePatchesSelect
    del negativePatchesSelect
print '############################################################################################'
print 'Generated outputs' 
print 'Patches for Dictionary - ', dictDataPointsPath
print 'Positive Patches for Training - ', positivePatchesSavePath
print 'Negative Patches for Training - ', negativePatchesSavePath
print 'Positive Patches for Test - ', positivePatchesTestSavePath
print 'Negative Patches for Test - ', negativePatchesTestSavePath
print '############################################################################################'
