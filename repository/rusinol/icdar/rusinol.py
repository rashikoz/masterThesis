import numpy as np
import os, sys
from numpy.lib.stride_tricks import as_strided
import vlfeat

# config parameters
# path realted configs
curPath = os.path.dirname(sys.argv[0]) 
# train data geneartion 

trainPath = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','train')
testPath = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','test')
resultPath = os.path.join('/data','rthalapp','results','rusinol','icdar')

# patch related
cellSize = (32,32)
stepSize = (1,1)
SIFT_DESC_DIM = 128

# key point detection ....Dense sampling
def denseSampler(imageArray, cellSizeDS, stepSizeDS=(1,1), verbose = 0):
    # numpy arrays used
    # imageArray - numpy array
    rowMax= imageArray.shape[0]
    colMax = imageArray.shape[1]    
    # using stride tips and tricks
    # get the row and column indices
    maxFeRows = np.floor((rowMax - cellSizeDS[0])/stepSizeDS[0]) + 1
    maxFeCols = np.floor((colMax - cellSizeDS[1])/stepSizeDS[0]) + 1
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
# Helper functions
########################################################################################
# represent each patch using SIFT descriptor
def getSIFTDescriptor(inputArray, patchSize):
    print 'getSIFTDescriptor - Sift Descriptor'
    print 'getSIFTDescriptor - Argument Array Shape - ', inputArray.shape
    siftArrayCollect = np.zeros((inputArray.shape[0], 128))    
    for patchRun in np.arange(inputArray.shape[0]):
        if not np.mod(patchRun, 10000):
            print  'Siftizing patches - ', patchRun, '/' , inputArray.shape[0]
        binSize = patchSize[0]/4
        eachStep = patchSize[0]
        frames, siftArray = vlfeat.vl_dsift(inputArray[patchRun].astype(np.float32), \
                                            size = binSize, step = eachStep)
        siftArrayCollect[patchRun] = siftArray.T
               
    print 'getSIFTDescriptor - Return Array Shape - ', siftArrayCollect.shape
    return(siftArrayCollect.astype(np.float32))

# represent each patch using SIFT descriptor
def siftizeDataPoints(inputArray, patchSize):
    print 'siftizeDataPoints - Sift Descriptor'
    print 'siftizeDataPoints - Argument Array Shape - ', inputArray.shape
    numOfPoints = inputArray.shape[0]
    pointPerbatch = np.minimum(20000, numOfPoints)    
    siftArrayCollect = np.array([])
    for stIndex in np.arange(0, numOfPoints, pointPerbatch):
        if not np.mod(stIndex, 50000):
            print  'Siftizing patches - ', stIndex, '/' , numOfPoints
        endIndex =  np.minimum(numOfPoints, stIndex + pointPerbatch)
        partPoints = np.reshape(inputArray[stIndex:endIndex],((endIndex - stIndex)*inputArray.shape[1], \
                                                              inputArray.shape[2]))
        partPoints = np.asarray(partPoints, dtype = np.float32)
        binSize = patchSize[0]/4
        eachStep = patchSize[0]  
        frames, siftArray = vlfeat.vl_dsift(partPoints, size = binSize, step = eachStep)
        if(siftArrayCollect.size == 0):
            siftArrayCollect = siftArray.T
        else:
            siftArrayCollect = np.vstack((siftArrayCollect, siftArray.T))
        stIndex += pointPerbatch
        
    print 'siftizeDataPoints - Return Array Shape - ', siftArrayCollect.shape
    return(siftArrayCollect.astype(np.float32))

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
