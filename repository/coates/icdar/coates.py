import numpy as np
import os
import sys
import time
from numpy.lib.stride_tricks import as_strided
from scipy.signal import fftconvolve
from scipy.sparse import coo_matrix
from scipy.misc import imresize, imread
import mserHelper

# config parameters
# path realted configs
curPath = os.path.dirname(sys.argv[0])
# train data geneartion 
trainPath = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','train')
testPath = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','test')
resultPath = os.path.join('/data','rthalapp','results','coates','icdar')

# patch related
cellSize = (32, 32)
stepSize = (1, 1)
subCellSize = (8, 8)
subStepSize = (1,1)
numPositiveTextPatches = 30000
numNegativeTextPatches = 120000
# spherical k means
numOfClusters = 32
numIterations = 500
constantZCA = 0.1

# key point detection ....Dense sampling
def denseSampler(imageArray, cellSizeDS, stepSizeDS=(1,1), verbose = 0):
    # numpy arrays used
    # imageArray - numpy array
    rowMax= imageArray.shape[0]
    colMax = imageArray.shape[1]
    # using stride tips and tricks
    # get the row and column indices
    maxFeRows = np.floor((rowMax - cellSizeDS[0])/stepSizeDS[0]) + 1
    maxFeCols = np.floor((colMax - cellSizeDS[1])/stepSizeDS[1]) + 1
    retArray = as_strided(imageArray, shape = (maxFeRows, maxFeCols, \
                                               cellSizeDS[0], cellSizeDS[1]), \
                                    strides = (stepSizeDS[0]*imageArray.strides[0], \
                                               stepSizeDS[1]*imageArray.strides[1], \
                                               imageArray.strides[0], imageArray.strides[1]))
    if verbose:
        print 'denseSampler - Cell Size - ', cellSizeDS, 'Step Size - ', stepSizeDS
        print 'denseSampler - Image Shape - ',imageArray.shape
        print 'denseSampler - Return Array Shape - ',retArray.shape
    return(retArray)

def imagePatchSampler(imageArray, cellSizeDS =(32,32), stepSizeDS = (5,5), cellSubSizeDS = (8,8), verbose = 0):
    # numpy arrays used
    # imageArray - numpy array
    rowMax= imageArray.shape[0]
    colMax = imageArray.shape[1]    
    # using stride tips and tricks
    # get the row and column indices
    maxFeRows = np.floor((rowMax - cellSizeDS[0])/stepSizeDS[0]) + 1
    maxFeCols = np.floor((colMax - cellSizeDS[1])/stepSizeDS[1]) + 1 
    rowSubNum = cellSizeDS[0] - cellSubSizeDS[0] + 1
    colSubNum = cellSizeDS[1] - cellSubSizeDS[1] + 1    
    retArray = as_strided(imageArray, shape = (maxFeRows, maxFeCols, rowSubNum,\
                                               colSubNum, cellSubSizeDS[0], cellSubSizeDS[1]), \
                                    strides = (stepSizeDS[0]*imageArray.strides[0], \
                                               stepSizeDS[1]*imageArray.strides[1], \
                                               1*imageArray.strides[0], \
                                               1*imageArray.strides[1], \
                                               imageArray.strides[0], \
                                               imageArray.strides[1])).astype(np.uint8)
    rows = retArray.shape[0]*retArray.shape[1]*retArray.shape[2]*retArray.shape[3]
    cols = retArray.shape[4]*retArray.shape[5]
    retArray = retArray.reshape((rows, cols))    
    if verbose:
        print 'imagePatchSampler - Cell Size - ', cellSizeDS, 'Step Size - ', stepSizeDS
        print 'imagePatchSampler - Image Shape - ',imageArray.shape
        print 'imagePatchSampler - Return Array Shape - ',retArray.shape
    return [retArray, maxFeRows, maxFeCols]

####################################################################################################
# identify the patch has text or not
def updateTextPresence(charInfos, imageSize, cellSizeTP, stepSizeTP):
    print 'updateTextPresence - Identifying the text patches'
    start = time.time()
    rowMax= imageSize[0]
    colMax = imageSize[1]
    maxFeRows = np.floor((rowMax - cellSizeTP[0])/stepSizeTP[0]) + 1
    maxFeCols = np.floor((colMax - cellSizeTP[1])/stepSizeTP[1]) + 1
    resultArray = np.zeros((maxFeRows, maxFeCols), dtype=np.int8)
    anyValidCharFlag, validCharBoxes = checkIfAnyValidChars(charInfos, cellSizeTP)
    if(anyValidCharFlag):
        arrIndex = np.ndindex(maxFeRows, maxFeCols)
        for indexRun in arrIndex:
            resultArray[indexRun] = testTextPresenceOpti(validCharBoxes, \
                                                        stepSizeTP[0]*indexRun[0], \
                                                        stepSizeTP[1]*indexRun[1], \
                                                        cellSizeTP)
    numTextPatches = len(np.where(resultArray == 1)[0])
    numNonTextPatches = len(np.where(resultArray == 0)[0])
    end = time.time()
    print 'updateTextPresence - Elapsed Time - ', end-start
    print 'updateTextPresence - Number of Text Patches - ', numTextPatches   
    print 'updateTextPresence - Number of Non Text Patches - ', numNonTextPatches 
    return(resultArray)

# text presence alogorithm
def getAllCharBoxesImage(taggedRectangles):
    boundingBoxChar = []
    boundingBoxWord = []
    for eachTagRect in taggedRectangles.getchildren(): 
        word = eachTagRect.getchildren()[0].text
        segDetails = eachTagRect.getchildren()[1].getchildren()
        eachTagRectDetails = eachTagRect.attrib
        coltag = int(float(eachTagRectDetails['x']))
        rowTag = int(float(eachTagRectDetails['y']))
        widthTag = int(float(eachTagRectDetails['width']))
        heightTag = int(float(eachTagRectDetails['height']))
        wordDetails = [rowTag, coltag, heightTag, widthTag]
        boundingBoxWord.append(wordDetails)
        charWidthArray = np.array([0])
        for tempRun in np.arange(len(segDetails)):
            charWidthArray = np.append(charWidthArray, int(float(segDetails[tempRun].text)))
        charWidthArray = np.append(charWidthArray, widthTag)
        charWidthArray = np.diff(charWidthArray)
        # run through the all the patches
        for eachCharRun in np.arange(len(word)):
            curChar = word[eachCharRun]
            rowCharTag = rowTag
            heightCharTag = heightTag
            widthCharTag = charWidthArray[eachCharRun]
            if eachCharRun == 0:
                colCharTag = coltag
            else:
                colCharTag += charWidthArray[eachCharRun-1]
            charDetails = [rowCharTag, colCharTag, heightCharTag, widthCharTag]
            boundingBoxChar.append(charDetails)
    print 'Number of characters in image - ', len(boundingBoxChar)
    print 'Number of words in image - ', len(boundingBoxWord)
    bboxCharArray = np.array(boundingBoxChar)
    bboxWordArray = np.array(boundingBoxWord)    
    return [bboxCharArray, bboxWordArray]

def checkIfAnyValidChars(bboxArray, cellSizeTC):
    if(bboxArray.shape[0] != 0):
        # range of character sizes 0.7*32 - 1.3*32
        maxCharBoxSizeHeight = np.ceil(1.3*cellSizeTC[0])
        minCharBoxSizeHeight = np.floor(0.7*cellSizeTC[0])
        maxCharBoxSizeWidth = np.ceil(1.3*cellSizeTC[1])
        minCharBoxSizeWidth = np.floor(0.7*cellSizeTC[1])
        #[rowCharTag, colCharTag, heightCharTag, widthCharTag]
        heightValues  = bboxArray[:,3] - bboxArray[:,1]
        widthValues  = bboxArray[:,2] - bboxArray[:,0]
        heightTrue = np.logical_and((heightValues <= maxCharBoxSizeHeight),(heightValues >= minCharBoxSizeHeight))
        widthTrue = np.logical_and((widthValues <= maxCharBoxSizeWidth),(widthValues >= minCharBoxSizeWidth))
        validCharIndex = np.logical_or(heightTrue, widthTrue)
        validIndex = np.where(validCharIndex == True)[0]
        validCharBoxesArray = bboxArray[validIndex,:]
        print 'Number of valid characters in image - ', validCharBoxesArray.shape[0]
        validCharStatusRetVal = np.any(validCharIndex) 
    else:
        # no text at all
        validCharStatusRetVal = False
        validCharBoxesArray = np.array([])
    return [validCharStatusRetVal, validCharBoxesArray]

def testTextPresenceOpti(bboxArray, eachRowCord, eachColCord, cellSizeTC):
    # convert to sets for intesection operation
    retVal = 0
    rowSetPatch = set(xrange(eachRowCord, eachRowCord+cellSizeTC[0])) 
    colSetPatch = set(xrange(eachColCord, eachColCord+cellSizeTC[1])) 
    patchArea = cellSizeTC[0]*cellSizeTC[1]
    numOfChars = bboxArray.shape[0]
    # running through the all tagged chars
    for eachCharRun in xrange(numOfChars): 
        rowCharTag = bboxArray[eachCharRun, 1]
        colCharTag = bboxArray[eachCharRun, 0]
        heightCharTag = bboxArray[eachCharRun,3] - bboxArray[eachCharRun,1]
        widthCharTag = bboxArray[eachCharRun,2] - bboxArray[eachCharRun,0]
        rowSetCharTag = set(xrange(rowCharTag, rowCharTag+heightCharTag)) 
        colSetCharTag = set(xrange(colCharTag, colCharTag+widthCharTag)) 
        charArea = widthCharTag*heightCharTag
        overlapRows = len(rowSetPatch.intersection(rowSetCharTag))
        overlapCols = len(colSetPatch.intersection(colSetCharTag))
        overlapArea = overlapRows*overlapCols
        overLapCharRatio = overlapArea/float(charArea)
        overLapPatchRatio = overlapArea/float(patchArea)
        if(overLapPatchRatio >= 0.8):
            # update the presence of text
            retVal = 1
    return(retVal)
####################################################################################################
# Normalization procedure
# subtract mean and divide by standard deviation    
# A small value is added to the variance before division to avoid divide by zero and
# also suppress noise. For pixel intensities in the range [0, 255], adding 10 to the
# variance is often a good starting point.
# inputArray  expects dimension of dimension x Number of data points
# each column of the array reprsents a datapoint
# matlab code from coates
# patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
def normalizeInput(inputArray,  varianceOffset = 10, verbose = 0): 
    #start = time.time()
    meanArray = np.mean(inputArray, 1, dtype= np.float64).reshape(-1,1)
    inputArray -= meanArray
    # x- mu / sigma    
    #inputArray = inputArray - np.tile(meanArray, (1, inputArray.shape[1])) 
    #inputArray = inputArray/np.tile(stdArray, (1, inputArray.shape[1]))
    stdDevArray = np.sqrt(np.var(inputArray, 1, dtype= np.float64, ddof = 1) + varianceOffset).reshape(-1,1)
    # stdDevArray[stdDevArray < 1e-8] = 1
    inputArray /= stdDevArray
    if verbose:
        print 'normalizeInput - Normalization'
        print 'normalizeInput - Argument Array Shape - ', inputArray.shape
        print 'normalizeInput - Return Array Shape - ', inputArray.shape 
    #end = time.time()
    #print 'normalizeInput - Elapsed Time - ', end-start                                             
    return(inputArray)

# A simple choice of whitening transform is the ZCA whitening transform. If
# VDV = cov(x) is the eigenvalue decomposition of the covariance of the data
# points x, then the whitened points are computed as , where
# zca is a small constant. For contrast-normalized data, setting zca to 0.01 for
# 16-by-16 pixel patches, or 0.1 for 8-by-8 pixel patches is a good starting point
# % ZCA whitening (with low-pass)
# C = cov(patches);
# M = mean(patches);
# [V,D] = eig(C);
# P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
# patches = bsxfun(@minus, patches, M) * P;
# xZCAwhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x
# sigma ..covariance matrix 
# sigma = np.dot(inputArray, inputArray.T) / inputArray.shape[1];
# [U,s,V] = np.linalg.svd(sigma);
# S = np.diag(s)   
# xZCAwhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x;
# whitenTransform = np.diag(1/(np.sqrt(np.diag(S) + epsilonZCA)))
# whitenTransform = np.dot(U, whitenTransform)
# whitenTransform = np.asarray(np.dot(whitenTransform, U.T), dtype = np.float32)        
# inputArray = np.dot(whitenTransform, inputArray)
def whitenInputsRowData(inputArray,  epsilonZCA):
    print 'whitenInputsRowData - ZCA whitening'
    print 'whitenInputsRowData - Argument Array Shape - ', inputArray.shape
    numOfPoints, dimension = inputArray.shape
    meanZCAArray = np.mean(inputArray, 0, dtype= np.float64).reshape(1,-1)
    sigma = np.dot(inputArray.T, inputArray) / float(numOfPoints)
    U, S, V = np.linalg.svd(sigma)
    whiteningMatrix = np.dot(U, np.diag(1.0/np.sqrt(S + epsilonZCA)))
    whiteningMatrix = np.dot(whiteningMatrix, U.T)
    inputArray -= meanZCAArray
    xZCAWhite = np.dot(inputArray, whiteningMatrix)
    zcaParams = [whiteningMatrix, meanZCAArray]
    print 'whitenInputsRowData - Return Array Shape - ', xZCAWhite.shape
    return([xZCAWhite, zcaParams])

def whitenInputsColumnData(inputArray,  epsilonZCA):
    print 'whitenInputsColumnData - ZCA whitening'
    print 'whitenInputsColumnData - Argument Array Shape - ', inputArray.shape
    # have saved all the data row wise
    inputArray = inputArray.T
    meanZCAArray = np.mean(inputArray, 1, dtype= np.float64).reshape(-1,1)
    dimension, numOfPoints = inputArray.shape
    sigma = np.dot(inputArray,inputArray.T) / float(numOfPoints)
    U, S, V = np.linalg.svd(sigma)
    whiteningMatrix = np.dot(U, np.diag(1.0/np.sqrt(S + epsilonZCA)))
    whiteningMatrix = np.dot(whiteningMatrix, U.T)
    inputArray -= meanZCAArray
    xZCAWhite = np.dot(whiteningMatrix, inputArray)
    xZCAWhite = xZCAWhite.T
    zcaParams = [whiteningMatrix, meanZCAArray]
    print 'whitenInputs - Return Array Shape - ', xZCAWhite.shape
    return([xZCAWhite, zcaParams])

# apply whiten
# patches = bsxfun(@minus, patches, M) * P;
def applyWhitenTransformRowData(inputArray, whitenParams, verbose = 0):
    # get the ZCA params for transform
    #numOfPoints, dimension = inputArray.shape
    inputArray -= whitenParams[1]
    inputArray = np.dot(inputArray, whitenParams[0])
    if verbose:
        print 'applyWhitenTransformRowData - ZCA whitening'
        print 'applyWhitenTransformRowData - Argument Array Shape - ', inputArray.shape
        print 'applyWhitenTransformRowData - Return Array Shape - ', inputArray.shape
    return(inputArray)

def applyWhitenTransformColumnData(inputArray, whitenParams, verbose = 0):
    # get the ZCA params for transform
    inputArray = inputArray.T
    #dimension, numOfPoints = inputArray.shape
    inputArray -= whitenParams[1]
    inputArray = np.dot(whitenParams[0], inputArray)
    inputArray = inputArray.T 
    if verbose:
        print 'applyWhitenTransformColumnData - ZCA whitening'
        print 'applyWhitenTransformColumnData - Argument Array Shape - ', inputArray.shape
        print 'applyWhitenTransformColumnData - Return Array Shape - ', inputArray.shape
    return(inputArray)

# preprocess the data for training #
def prepareData(inputArray, epsilonZCA):
    print 'prepareData - Preparing the data'
    print 'prepareData - Argument Array Shape - ', inputArray.shape
    # normalisation
    inputArray = normalizeInput(inputArray.astype(np.float64))
    print 'Min - ',np.min(inputArray)
    print 'Max - ',np.max(inputArray)
    print 'Mean - ',np.mean(inputArray)
    print 'Std Dev - ',np.std(inputArray)
    # whitening
    print 'Eigen values before whitening'
    covInputData =  np.dot(inputArray.T,inputArray) / inputArray.shape[0]
    print np.sort(abs(np.linalg.eig(covInputData)[0]))
    inputArray, zcaParamsList = whitenInputsRowData(inputArray, epsilonZCA)
    print 'After Whitening - epsilon - ', epsilonZCA
    print 'Min - ',np.min(inputArray)
    print 'Max - ',np.max(inputArray)
    print 'Mean - ',np.mean(inputArray)
    print 'Std Dev - ',np.std(inputArray)
    # save all the parems for later use
    print 'prepareData - Return Array Shape - ', inputArray.shape
    return([inputArray, zcaParamsList])
####################################################################################################
# clustering spherical k means
# implemented as per Learning Feature Representations with K-means
# Adam Coates and Andrew Y. Ng
def sphericalKmeans(dataToBeClustered, numOfClustersSKM, numIterationsSKM, batchSize = 10000):
    start = time.time()
    print 'sphericalKmeans - Spherical K Means'
    print 'sphericalKmeans - Data Array Shape - ', dataToBeClustered.shape
    numOfDataPoints, dimension = dataToBeClustered.shape
    dictionary = np.random.randn(numOfClustersSKM, dimension)
    vecLength = np.sqrt(np.sum((dictionary**2), 1)).reshape(-1, 1)
    #dictionary = dictionary/np.tile(vecLength, (1, dictionary.shape[1]))
    dictionary /= vecLength   
    clusterAssignment = np.zeros((numOfDataPoints, 1))
    for iRun in np.arange(numIterationsSKM):
        print 'iteration - ', iRun ,'/', numIterationsSKM
        #summation = zeros(k, size(X,2));
        summation = np.zeros((numOfClustersSKM, dimension))
        #counts = zeros(k, 1);
        counts = np.zeros((numOfClustersSKM, 1))
        for batchRun in np.arange(0, numOfDataPoints, batchSize):
            curBatch = dataToBeClustered[batchRun:batchRun+batchSize,:]
            numPointsInBatch = curBatch.shape[0]
            projections = np.dot(dictionary, curBatch.T)
            maxIndex = np.argmax(abs(projections), 0)
            clusterAssignment[batchRun:batchRun+numPointsInBatch,0] = maxIndex
            cols = np.arange(numPointsInBatch)
            dataOne = np.ones((numPointsInBatch))
            sparseMat = coo_matrix((dataOne,(maxIndex,cols)), shape=(numOfClustersSKM, numPointsInBatch))
            sArray = np.multiply(projections, sparseMat.toarray())
            summation = summation + np.dot(sArray, curBatch)
            counts = counts + np.sum(sArray, 1)
        vecLength = np.sqrt(np.sum((summation**2), 1)).reshape(-1, 1)
        #dictionary = summation/np.tile(vecLength, (1, summation.shape[1]))
        dictionary = summation/vecLength  
        # find the empty clusters
        badClusterIds = np.where(counts==0)[0]
        if(badClusterIds.size != 0):
            print 'Warning bad clusters'
            dictionary[badClusterIds,:] = 0 
    end = time.time()
    print 'sphericalKmeans - Elapsed Time - ', end-start 
    print 'sphericalKmeans - Dictionary Shape - ', dictionary.shape
    return dictionary

def selectCenteroids(dictionary , varThreshold = 0.025):
    #remove centroids whos variance is lower than varthresh
    print 'selectCenteroids - Argument Array Shape - ', dictionary.shape
    #varD = np.zeros((dictionary.shape[0], 1))
    newDictionary = np.array([])
    for dictRun in np.arange(dictionary.shape[0]):
        eachCenteroid = dictionary[dictRun]
        curVar = np.var(eachCenteroid - np.min(eachCenteroid))/(np.max(eachCenteroid)-np.min(eachCenteroid))
        if curVar > varThreshold:
            if(newDictionary.size == 0):
                newDictionary = eachCenteroid
            else:    
                newDictionary = np.vstack((newDictionary, eachCenteroid))
        else:
            print 'Variance less than threshold - ',  dictRun
        
    print 'selectCenteroids - No of centeroids removed - ', dictionary.shape[0] - newDictionary.shape[0]        
    print 'selectCenteroids - Return Array Shape - ', newDictionary.shape              
    return newDictionary    

###################################################################################################
def standardizeFeatures(inputArray, feStdParams=0, verbose = 0):
    if(feStdParams == 0):
        meanAvgVal = np.mean(inputArray, 0, dtype= np.float64).reshape(1,-1)
        stdAvgVal = np.sqrt(np.var(inputArray, 0, dtype= np.float64, ddof = 1) + 0.01).reshape(1,-1)
        featureParams = [meanAvgVal, stdAvgVal]
    else:
        featureParams = feStdParams
        meanAvgVal = feStdParams[0]
        stdAvgVal = feStdParams[1] 
    # standardized       
    inputArray -= meanAvgVal
    inputArray /= stdAvgVal
    if verbose:
        print 'standardizeFeatures - feature standardize'
        print 'standardizeFeatures - Argument Array Shape - ', inputArray.shape  
        print 'standardizeFeatures - Return Array Shape - ', inputArray.shape
    return([inputArray, featureParams])
###################################################################################################
def getQuadIndices(numSubRow, numSubCol, poolShape):
    #numPools = poolShape[0]*poolShape[1]
    numSubPatches = numSubCol*numSubRow
    indexArray = np.arange(numSubPatches).reshape(numSubRow, numSubCol)
    quadIndices = []
    rowSplitArray = np.array_split(indexArray, [numSubRow/3, 2*(numSubRow/3)], 0)
    for iRun in np.arange(len(rowSplitArray)):
        eachRowSplitArray = rowSplitArray[iRun]
        # split the array into 3 columns
        columnSplitArray = np.array_split(eachRowSplitArray, [numSubCol/3, 2*(numSubCol/3)], 1)
        for jRun in np.arange(len(columnSplitArray)):
            quadIndices.append(columnSplitArray[jRun])
    return(quadIndices)

# patches feature generation
def computeResponsePatches(patchArray, dictionary, patchSizeTraining, \
                     subCellSizeCR, subStepSizeCR, trainingParamsList, verbose = 0):
    # batch size
    start = time.time()
    batchSize = 200
    numOfClustersCR = dictionary.shape[0] 
    numSubRow = patchSizeTraining[0] - subCellSizeCR[0] + 1
    numSubCol = patchSizeTraining[1] - subCellSizeCR[1] + 1
    numSubPatches = numSubCol*numSubRow
    dimension = subCellSizeCR[0]*subCellSizeCR[1]
    fpool = (3,3)
    numFtiles = fpool[0]*fpool[1]
    # get the quad indices
    quadIndices = getQuadIndices(numSubRow, numSubCol, fpool)
    featureComplete = np.zeros((patchArray.shape[0], numFtiles*numOfClustersCR))
    for batchRun in np.arange(0, patchArray.shape[0], batchSize):
        if not np.mod(batchRun, 5000) or (batchRun== patchArray.shape[0]-batchSize):
            if verbose: 
                print 'Extracting features - ', batchRun, '/', patchArray.shape[0]
        curBatch = patchArray[batchRun:batchRun+batchSize,:,:]
        numOfitems = curBatch.shape[0]
        # patches
        patchSelect = np.zeros((numOfitems*numSubPatches, dimension),dtype = np.uint8)
        for eachPatchRun in np.arange(numOfitems):
            subPatchArray = denseSampler(curBatch[eachPatchRun,:,:], subCellSizeCR, subStepSizeCR)
            subPatchArray = np.reshape(subPatchArray,(subPatchArray.shape[0]*subPatchArray.shape[1],\
                                                      subPatchArray.shape[2]*subPatchArray.shape[3]))
            patchSelect[eachPatchRun*numSubPatches:numSubPatches*(eachPatchRun+1),:] = subPatchArray
                       
        curBatch = normalizeInput(patchSelect.astype(np.float64))
        del patchSelect
        curBatch = applyWhitenTransformRowData(curBatch, trainingParamsList)
        #curBatch = np.dot(dictionary, curBatch.T).T
        curBatch = np.dot(curBatch, dictionary.T)
        curBatch = np.abs(curBatch) - 0.5 
        curBatch[curBatch < 0] = 0
        # pooling            
        for eachPatchRun in np.arange(numOfitems):
            addIndex = numSubPatches*eachPatchRun
            # run through all the quadrants
            for poolRun in np.arange(len(quadIndices)):
                finalPatchRun = eachPatchRun + batchRun
                curQuad = quadIndices[poolRun].ravel() + addIndex
                # update the feature values
                featureComplete[finalPatchRun,\
                (poolRun*numOfClustersCR):((poolRun+1)*numOfClustersCR)] = np.sum(curBatch[curQuad,:], 0)
                curBatch[curQuad,:] = 0
    end = time.time()
    if verbose:
        print 'computeResponsePatches - Computing features for patches'
        print 'computeResponsePatches - Argument Array Shape - ', patchArray.shape
        print 'computeResponsePatches - Elapsed Time in seconds - ', end-start
        print 'computeResponsePatches - Return Array Shape - ', featureComplete.shape
    return(featureComplete)

# image feature generation
def computeResponseImage(curImage, dictionary, patchSizeTraining, patchStepTraining, subCellSizeCR, \
                          subStepSizeCR, trainingParamsList, svmParams):
    # batch size
    start = time.time()
    fpool = (3,3)
    numFtiles = fpool[0]*fpool[1]
    batchSize = 200
    patchBatchSize = 200
    maxRowImage = curImage.shape[0]
    maxColImage = curImage.shape[1]
    numOfClustersCR = dictionary.shape[0]
    numSubRow = patchSizeTraining[0] - subCellSizeCR[0] + 1
    numSubCol = patchSizeTraining[1] - subCellSizeCR[1] + 1
    numSubPatches = numSubCol*numSubRow
    quadIndices = getQuadIndices(numSubRow, numSubCol, fpool)
    maxFeRows = np.floor((maxRowImage - patchSizeTraining[0])/patchStepTraining[0]) + 1
    maxFeCols = np.floor((maxColImage - patchSizeTraining[1])/patchStepTraining[1]) + 1
    maxFeRowsBatch = 1 + np.floor((batchSize-patchSizeTraining[0])/patchStepTraining[0])
    maxFeColsBatch = 1 + np.floor((batchSize-patchSizeTraining[1])/patchStepTraining[1])
    txtPredictions = np.zeros((maxFeRows, maxFeCols))
    zcaParams = [trainingParamsList[0],trainingParamsList[1]]
    meanValFeatures = trainingParamsList[2]
    stdValFeatures = trainingParamsList[3]
    rowFeRun = 0 
    for rowImageRun in xrange(0, int(maxFeRows*patchStepTraining[0] + 1), int(maxFeRowsBatch*patchStepTraining[0])):
        colFeRun = 0
        for colImageRun in xrange(0, int(maxFeCols*patchStepTraining[1] + 1), int(maxFeColsBatch*patchStepTraining[1])):
            curImagePatch = curImage[rowImageRun:rowImageRun+batchSize, colImageRun:colImageRun+batchSize]
            patchCollectArray, numRowPatches, numColPatches = imagePatchSampler(curImagePatch, patchSizeTraining, 
                                                  patchStepTraining, subCellSizeCR)   
            txtPredictionsPatch = np.zeros((numRowPatches*numColPatches, 1))
            finalPatchRun = 0
            for batchRun in xrange(0, patchCollectArray.shape[0], int(numSubPatches*patchBatchSize)):
                curBatch = patchCollectArray[batchRun:batchRun+(numSubPatches*patchBatchSize),:]
                numOfPatches = curBatch.shape[0]/numSubPatches
                curBatch = normalizeInput(curBatch.astype(np.float64))
                curBatch = applyWhitenTransformRowData(curBatch, zcaParams)
                curBatch = np.dot(curBatch, dictionary.T)
                curBatch = np.abs(curBatch)- 0.5
                curBatch[curBatch < 0] = 0
                # run through all patches
                for eachPatchRun in xrange(numOfPatches):
                    addIndex = numSubPatches*eachPatchRun
                    featurePatch = np.zeros((1, (numFtiles*numOfClustersCR)))
                    # run through all the quadrants
                    for poolRun in np.arange(len(quadIndices)):
                        curQuad = quadIndices[poolRun].ravel() + addIndex
                        # update the feature values
                        featurePatch[0, (poolRun*numOfClustersCR):((poolRun+1)*numOfClustersCR)] = np.sum(curBatch[curQuad,:], 0)
                    featurePatch -= meanValFeatures
                    featurePatch /= stdValFeatures
                    featurePatch = np.append(featurePatch, 1).reshape(1,-1)
                    txtPredictionsPatch[finalPatchRun, 0] = np.dot(featurePatch, svmParams)[0,0]
                    finalPatchRun += 1 
            txtPredictions[rowFeRun:rowFeRun+numRowPatches, colFeRun:colFeRun+numColPatches] = \
                    txtPredictionsPatch.reshape((numRowPatches, numColPatches))
            colFeRun += numColPatches
        rowFeRun += numRowPatches
    end = time.time()
    print 'computeResponseImage - Elapsed Time in seconds - ', end-start 
    print 'computeResponseImage - Return Array Shape - ', txtPredictions.shape
    return(txtPredictions)

def computeResponseImageAll(curImage, dictionary, patchSizeTraining, patchStepTraining, subCellSizeCR, \
                          subStepSizeCR, trainingParamsList, svmParams):
    # batch size
    start = time.time()
    fpool = (3,3)
    numFtiles = fpool[0]*fpool[1]
    patchBatchSize = 200
    maxRowImage = curImage.shape[0]
    maxColImage = curImage.shape[1]
    numOfClustersCR = dictionary.shape[0]
    numSubRow = patchSizeTraining[0] - subCellSizeCR[0] + 1
    numSubCol = patchSizeTraining[1] - subCellSizeCR[1] + 1
    numSubPatches = numSubCol*numSubRow
    quadIndices = getQuadIndices(numSubRow, numSubCol, fpool)  
    maxFeRows = np.floor((maxRowImage - patchSizeTraining[0])/patchStepTraining[0]) + 1
    maxFeCols = np.floor((maxColImage - patchSizeTraining[1])/patchStepTraining[0]) + 1
    txtPredictions = np.zeros((maxFeRows*maxFeCols, 1)) 
    zcaParams = [trainingParamsList[0],trainingParamsList[1]]
    meanValFeatures = trainingParamsList[2]
    stdValFeatures = trainingParamsList[3]
    patchCollectArray, numRowPatches, numColPatches = imagePatchSampler(curImage, patchSizeTraining, 
                                                  patchStepTraining, subCellSizeCR)   
    for batchRun in xrange(0, patchCollectArray.shape[0], int(numSubPatches*patchBatchSize)):
        curBatch = patchCollectArray[batchRun:batchRun+(numSubPatches*patchBatchSize),:]
        numOfPatches = curBatch.shape[0]/numSubPatches
        curBatch = normalizeInput(curBatch.astype(np.float64))
        curBatch = applyWhitenTransformRowData(curBatch, zcaParams)
        curBatch = np.dot(curBatch, dictionary.T)
        curBatch = np.abs(curBatch)- 0.5
        curBatch[curBatch < 0] = 0
        # run through all patches
        for eachPatchRun in xrange(numOfPatches):
            addIndex = numSubPatches*eachPatchRun
            finalPatchRun = eachPatchRun + (batchRun/numSubPatches)
            # run through all the quadrants
            featurePatch = np.zeros((1, (numFtiles*numOfClustersCR)))
            featurePatch[0,-1] = 1
            for poolRun in np.arange(len(quadIndices)):
                curQuad = quadIndices[poolRun].ravel() + addIndex
                # update the feature values
                featurePatch[0 ,(poolRun*numOfClustersCR):((poolRun+1)*numOfClustersCR)] = np.sum(curBatch[curQuad,:], 0)
            featurePatch -= meanValFeatures
            featurePatch /= stdValFeatures
            featurePatch = np.append(featurePatch, 1).reshape(1,-1)
            txtPredictions[finalPatchRun, 0] = np.dot(featurePatch, svmParams)[0,0]
    # convert to svm predictions
    txtPredictions = txtPredictions.reshape(maxFeRows, maxFeCols) 
    end = time.time()
    print 'computeResponseImage - Elapsed Time in seconds - ', end-start 
    print 'computeResponseImage - Return Array Shape - ', txtPredictions.shape
    return(txtPredictions)
####################################################################################################
# convolution feature extraction
# find features stride 1 (always)
def correlateWithKernel(imgArray, kernel, mode='valid'):
    return fftconvolve(imgArray, np.flipud(np.fliplr(kernel)), mode='valid')

def getFeaturesThroughConvolution(imageArray, dictionary, patchSizeTraining, subCellSizeCR, \
                                  trainingParamsList, verbose = 0): 
    # stride is always one
    start = time.time()
    fpool = (3,3)
    numFtiles = fpool[0]*fpool[1]
    numOfClustersCR = dictionary.shape[0]
    filterDim = np.sqrt(dictionary.shape[1])
    numSubRow = patchSizeTraining[0] - subCellSizeCR[0] + 1
    numSubCol = patchSizeTraining[1] - subCellSizeCR[1] + 1
    #numSubPatches = numSubCol*numSubRow
    quadIndices = getQuadIndices(numSubRow, numSubCol, fpool)
    maxRowImage = imageArray.shape[0]
    maxColImage = imageArray.shape[1]
    maxFeaureArrayRow = maxRowImage - filterDim + 1 
    maxFeaureArrayCol = maxColImage - filterDim + 1
    # pre processing 
    numElementsinFilter = filterDim*filterDim
    imageArray = imageArray.astype(np.float64)
    avgFilter = np.ones((filterDim, filterDim))/(numElementsinFilter)
    meanPatch = correlateWithKernel(imageArray, avgFilter, mode='valid')
    varPatch = correlateWithKernel(imageArray**2, avgFilter, mode='valid') - meanPatch**2
    # done to match matlab var function
    stdPatch = np.sqrt(((varPatch*numElementsinFilter)/(numElementsinFilter-1)) + 10)
    whiteningArray =  trainingParamsList[0]
    meanValZCA = trainingParamsList[1]
    dictionary = np.dot(dictionary, whiteningArray)
    # array for storage
    pooledfeatureValues = np.zeros((maxFeaureArrayRow-numSubRow+1, maxFeaureArrayCol-numSubCol+1, \
                                                                   numFtiles*numOfClustersCR))
    for filterRun in np.arange(numOfClustersCR):
        curFilter = dictionary[filterRun,:].reshape(1,-1)
        learnedfilter = curFilter.reshape(filterDim, filterDim)
        convResponse = correlateWithKernel(imageArray, learnedfilter, 'valid')
        convResponse -= (np.sum(curFilter)*meanPatch)
        convResponse /= (stdPatch)
        convResponse -= np.dot(curFilter, meanValZCA.T)
        #encoder
        convResponse = np.abs(convResponse) - 0.5
        convResponse[convResponse < 0] = 0
        # average pooling
        for poolRun in np.arange(len(quadIndices)):
            poolFilter = np.zeros((numSubRow*numSubCol, 1))
            curQuad = quadIndices[poolRun].ravel()
            poolFilter[curQuad, 0] = 1
            poolFilter = poolFilter.reshape(numSubRow, numSubCol)
            curPointer = poolRun*numOfClustersCR + filterRun
            pooledfeatureValues[:,:, curPointer] = correlateWithKernel(convResponse, poolFilter, 'valid')
    end = time.time()
    if verbose:
        print 'getFeaturesThroughConvolution - Elapsed Time in seconds - ', end-start 
        print 'getFeaturesThroughConvolution - Return Array Shape - ', pooledfeatureValues.shape
    return pooledfeatureValues

# for images   
def computeResponseImageConvolution(curImage, dictionary, patchSizeTraining, subCellSizeCR, \
                                    trainingParamsList, svmParams):
    # stride is always one
    start = time.time()
    batchSize = 500
    maxRowImage = curImage.shape[0]
    maxColImage = curImage.shape[1]
    maxFeRows = maxRowImage - patchSizeTraining[0] +1
    maxFeCols = maxColImage - patchSizeTraining[1] +1
    txtPredictions = np.zeros((maxFeRows, maxFeCols))    
    meanFeatureValues =  trainingParamsList[2]
    stdDevFeatureValues =  trainingParamsList[3]
    for rowImageRun in np.arange(0, maxFeRows, 1+batchSize-patchSizeTraining[0]):
        for colImageRun in np.arange(0, maxFeCols, 1+batchSize-patchSizeTraining[1]):            
            curImagePatch = curImage[rowImageRun:rowImageRun+batchSize, colImageRun:colImageRun+batchSize]
            curImageRowIndex = curImagePatch.shape[0] - patchSizeTraining[0] + 1
            curImageColIndex = curImagePatch.shape[1] - patchSizeTraining[1] + 1 
            feValue = getFeaturesThroughConvolution(curImagePatch, dictionary, patchSizeTraining, \
                                                            subCellSizeCR, trainingParamsList)
            feValue = feValue.reshape(curImageRowIndex*curImageColIndex, -1)
            feValue -= meanFeatureValues
            feValue /= stdDevFeatureValues
            feValue = np.hstack((feValue, np.ones((feValue.shape[0], 1))))
            predict = np.dot(feValue, svmParams)
            predictPart = predict[:,0].reshape(curImageRowIndex, curImageColIndex)
            txtPredictions[rowImageRun:rowImageRun+curImageRowIndex,\
                                    colImageRun:colImageRun+curImageColIndex] = predictPart
    end = time.time()
    print 'computeResponseImageConvolution - Elapsed Time in seconds - ', end-start 
    print 'computeResponseImageConvolution - Return Array Shape - ', txtPredictions.shape
    return(txtPredictions)

# batch wise implementation for features
def computeResponseImageConvolutionBatch(curImage, dictionary, patchSizeTraining, subCellSizeCR, \
                                    trainingParamsList, svmParams):
    # stride is always one
    start = time.time()
    batchSize = 200
    featureBatch = 100
    fpool = (3,3)
    numFtiles = fpool[0]*fpool[1]
    maxNumFeatures = dictionary.shape[0]
    maxRowImage = curImage.shape[0]
    maxColImage = curImage.shape[1]
    maxFeRows = maxRowImage - patchSizeTraining[0] +1
    maxFeCols = maxColImage - patchSizeTraining[1] +1
    txtPredictions = np.zeros((maxFeRows, maxFeCols))
    meanFeatureValues =  trainingParamsList[2]
    stdDevFeatureValues =  trainingParamsList[3]
    for rowImageRun in np.arange(0, maxFeRows, 1+batchSize-patchSizeTraining[0]):
        for colImageRun in np.arange(0, maxFeCols, 1+batchSize-patchSizeTraining[1]):
            curImagePatch = curImage[rowImageRun:rowImageRun+batchSize, colImageRun:colImageRun+batchSize]
            curImageRowIndex = curImagePatch.shape[0] - patchSizeTraining[0] + 1
            curImageColIndex = curImagePatch.shape[1] - patchSizeTraining[1] + 1
            for featureRun in np.arange(0, maxNumFeatures,featureBatch):
                curDictionary = dictionary[featureRun:featureRun+featureBatch,:]
                numFeaturesInCurBatch = curDictionary.shape[0]
                curMeanValue = np.zeros((1, numFtiles*numFeaturesInCurBatch))
                curStdDevValue = np.ones((1, numFtiles*numFeaturesInCurBatch))
                curSVMParams = np.zeros((numFtiles*numFeaturesInCurBatch, svmParams.shape[1]))
                for poolRun in np.arange(numFtiles):
                    srcPointerStart = (poolRun*maxNumFeatures)+featureRun
                    destPointerStart = poolRun*numFeaturesInCurBatch
                    if(meanFeatureValues.ndim == 2) and(stdDevFeatureValues.ndim == 2) :
                        curMeanValue[0, destPointerStart: destPointerStart+numFeaturesInCurBatch] = \
                            meanFeatureValues[0, srcPointerStart:srcPointerStart+numFeaturesInCurBatch]
                        curStdDevValue[0, destPointerStart: destPointerStart+numFeaturesInCurBatch] = \
                            stdDevFeatureValues[0, srcPointerStart:srcPointerStart+numFeaturesInCurBatch]
                    curSVMParams[destPointerStart: destPointerStart+numFeaturesInCurBatch,:] = \
                                 svmParams[srcPointerStart:srcPointerStart+numFeaturesInCurBatch,:]
                feValue = getFeaturesThroughConvolution(curImagePatch, curDictionary, patchSizeTraining, \
                                                                subCellSizeCR, trainingParamsList)
                feValue = feValue.reshape(curImageRowIndex*curImageColIndex, -1)
                feValue -= curMeanValue
                feValue /= curStdDevValue
                predict = np.dot(feValue, curSVMParams)
                predictPart = predict[:,0].reshape(curImageRowIndex, curImageColIndex)
                txtPredictions[rowImageRun:rowImageRun+curImageRowIndex,\
                                        colImageRun:colImageRun+curImageColIndex] += predictPart
            txtPredictions[rowImageRun:rowImageRun+curImageRowIndex,\
                                        colImageRun:colImageRun+curImageColIndex] += svmParams[-1,0]
    end = time.time()
    print 'computeResponseImageConvolution - Elapsed Time in seconds - ', end-start 
    print 'computeResponseImageConvolution - Return Array Shape - ', txtPredictions.shape
    return(txtPredictions)

# for patches
def computeResponsePatchesConvolution(patchArray, dictionary, patchSizeTraining, \
                     subCellSizeCR, trainingParamsList, verbose = 0):
    start = time.time()
    numOfClustersCR = dictionary.shape[0] 
    fpool = (3,3)
    numFtiles = fpool[0]*fpool[1]
    featureComplete = np.zeros((patchArray.shape[0], numFtiles*numOfClustersCR))
    for patchRun in np.arange(0, patchArray.shape[0]):
        if not np.mod(patchRun, 5000) and verbose:
            print 'Extracting features - ', patchRun, '/', patchArray.shape[0]
        curPatch =  patchArray[patchRun,:,:]      
        fevals = getFeaturesThroughConvolution(curPatch, dictionary, patchSizeTraining, \
                                                                subCellSizeCR, trainingParamsList)
        featureComplete[patchRun,:] = fevals[0,0,:]
    end = time.time()
    if verbose:
        print 'computeResponsePatchesConvolution - Argument Array Shape - ', patchArray.shape
        print 'computeResponsePatchesConvolution - Elapsed Time in seconds - ', end-start
        print 'computeResponsePatchesConvolution - Return Array Shape - ', featureComplete.shape
    return featureComplete
###################################################################################################       
# final max prediction in one function
def getFinalPrediction(dictionary, patchSizeTraining, patchStepTraining, \
                                   patchSizeDictionary, patchStepDictionary, \
                                   zcaParams , scaleList, testPredictFolder,\
                                   svmWeights, imgPathList): 
    imgRun = 0
    for imgPath in imgPathList:
        start = time.time()
        imgRun += 1 
        dirName, fName = os.path.split(imgPath)
        fName = fName.split('.')[0]
        dirName = dirName.split('/')[-1]
        print 'getFinalPrediction - Starting processing Image - ', imgPath, ' - ', imgRun , '/', len(imgPathList)
        # save file names
        predictFileName = os.path.join(testPredictFolder, 'MaxPredict_' + dirName + '_' + fName + '.npy')
        minT  = 0
        if(os.path.isfile(predictFileName)):
            print 'getFinalPrediction - Data already present'
            imageArray = np.round(imread(imgPath, flatten=True)).astype(np.uint8)            
        else:
            # open the image and load the image as array
            imageArray = np.round(imread(imgPath, flatten=True)).astype(np.uint8)
            maxPredictImage = np.ones(imageArray.shape)*(-np.Inf)
            for scaleRun in xrange(len(scaleList)):
                curScale = scaleList[scaleRun]
                curScale =0.5
                print 'getFinalPrediction - Computing the response for image ', fName, ' at scale ', curScale
                # rescale the image based on ratio given
                rescaledImage = imresize(imageArray, curScale, interp = 'bicubic')
                # features
                if((rescaledImage.shape[0] >= 32) and (rescaledImage.shape[1] >=32)):
                    if (patchStepTraining == (1,1)):
                        # feature convolution extraction
                        txtPredictions = computeResponseImageConvolutionBatch(rescaledImage, dictionary, \
                                        patchSizeTraining, patchSizeDictionary, zcaParams, svmWeights)
                        #print np.allclose(txtPredictions, txtPredictionsbatch)
                    else:
                        # feature extraction patch extraction           
                        txtPredictions = computeResponseImage(rescaledImage, dictionary, patchSizeTraining, \
                                        patchStepTraining, patchSizeDictionary, patchStepDictionary, \
                                        zcaParams, svmWeights)
                    arrIndex = np.ndindex(txtPredictions.shape[0], txtPredictions.shape[1])
                    for predictRun in arrIndex:
                        eachPredict = txtPredictions[predictRun[0], predictRun[1]]
                        minT = np.minimum(minT, eachPredict)
                        eachRowCord = int(np.floor((predictRun[0]*patchStepTraining[0])/curScale))
                        eachColCord = int(np.floor((predictRun[1]*patchStepTraining[1])/curScale))
                        eachRowCordEnd = int(eachRowCord + np.floor(patchSizeTraining[0]/curScale))
                        eachColCordEnd = int(eachColCord + np.floor(patchSizeTraining[1]/curScale))
                        predPart = maxPredictImage[eachRowCord:eachRowCordEnd, eachColCord:eachColCordEnd]
                        maxPredictImage[eachRowCord:eachRowCordEnd, eachColCord:eachColCordEnd] = np.maximum(predPart, eachPredict)
            # save the response
            maxPredictImage[maxPredictImage==-np.Inf] = minT
            np.save(predictFileName, maxPredictImage)
        end = time.time()
        print  'getFinalPrediction - Finished processing  - ', imgPath, ' seconds - ', end-start
        print '############################################################################################'
       
# final max prediction in one function
def getFinalPredictionMSER(dictionary, patchSizeTraining, patchStepTraining, \
                                   patchSizeDictionary, patchStepDictionary, \
                                   zcaParams , testPredictFolder,\
                                   svmWeights, imgPathList): 
    imgRun = 0
    for imgPath in imgPathList:
        start = time.time()
        imgRun += 1 
        dirName, fName = os.path.split(imgPath)
        fName = fName.split('.')[0]
        dirName = dirName.split('/')[-1]
        print 'getFinalPredictionMSER - Starting processing Image - ', imgPath, ' - ', imgRun , '/', len(imgPathList)
        # save file names
        predictFileName = os.path.join(testPredictFolder, 'MaxPredict_' + dirName + '_' + fName + '.npy')
        minT  = 0
        if(os.path.isfile(predictFileName)):
            print 'getFinalPredictionMSER - Data already present'
        else:
            # open the image and load the image as array
            imageArray = np.round(imread(imgPath, flatten=True)).astype(np.uint8)
            orgImageSize = imageArray.shape
            delta = 5
            minArea = 30
            maxArea = 90000
            maxVariation = 0.2
            minDiversity = 0.1
            # 1 column - width 2nd column height
            bboxesDetected = mserHelper.detectMSERBboxes(imgPath, delta, minArea, 
                                                         maxArea, maxVariation, minDiversity)
            numOfScales = bboxesDetected.shape[0]
            print 'getFinalPredictionMSER - Number of scales detected by MSER  in image - ', fName, ' - ', numOfScales
            if(numOfScales == 0):
                bboxesDetected = [patchSizeTraining[0],patchSizeTraining[1]]
            maxPredictImage = np.ones(imageArray.shape)*(-np.Inf)
            for scaleRun in xrange(numOfScales):
                curBBoxes = bboxesDetected[scaleRun,:]
                widthRatio = float(patchSizeTraining[1])/curBBoxes[0]
                heightRatio = float(patchSizeTraining[0])/curBBoxes[1]
                rescaledImgSize = (int(orgImageSize[0]*heightRatio), int(orgImageSize[1]*widthRatio))
                rescaledImage = imresize(imageArray, rescaledImgSize, interp = 'bicubic')
                print 'getFinalPredictionMSER - Computing the response for image ', fName, ' size ', rescaledImage.shape
                # features
                if((rescaledImage.shape[0] >= 32) and (rescaledImage.shape[1] >=32)):
                    if (patchStepTraining == (1,1)):
                        # feature convolution extraction
                        txtPredictions = computeResponseImageConvolutionBatch(rescaledImage, dictionary, \
                                        patchSizeTraining, patchSizeDictionary, zcaParams, svmWeights)
                        #print np.allclose(txtPredictions, txtPredictionsbatch)
                    else:
                        # feature extraction patch extraction           
                        txtPredictions = computeResponseImage(rescaledImage, dictionary, patchSizeTraining, \
                                        patchStepTraining, patchSizeDictionary, patchStepDictionary, \
                                        zcaParams, svmWeights)
                        #print np.allclose(txtPredictionsP, txtPredictions)
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
            # save the response
            maxPredictImage[maxPredictImage==-np.Inf] = minT
            np.save(predictFileName, maxPredictImage)
        end = time.time()
        print  'getFinalPredictionMSER - Finished processing  - ', imgPath, ' seconds - ', end-start
        print '############################################################################################'
