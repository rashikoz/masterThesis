import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn.cluster import k_means
import matplotlib.cm as cm

def detectMSERBboxes(imagePath, delta , minArea, maxArea, maxVariation, minDiversity, plotProgress = False):
    aspectRatioMin = 0.1
    aspectRatioMax = 2.5
    heightMin = 10
    heightMax = 400
    widthMin = 5
    widthMax = 300
    minNumClusters = 14
    dirName, fName = os.path.split(imagePath)
    imageOrg = cv.imread(imagePath)
    imageArray = cv.imread(imagePath, cv.CV_LOAD_IMAGE_GRAYSCALE)   
    # mser open cv
    bboxes = []
    mser = cv.MSER(delta, minArea, maxArea, maxVariation, minDiversity)
    regions = mser.detect(imageArray, None)
    for eachRegion in  xrange(len(regions)):
        eachMSERRegion = regions[eachRegion]
        minXCordinate = np.min(eachMSERRegion[:,0])
        maxXCordinate = np.max(eachMSERRegion[:,0])
        minYCordinate = np.min(eachMSERRegion[:,1])
        maxYCordinate = np.max(eachMSERRegion[:,1])
        leftTopCorner = (minXCordinate, minYCordinate)
        bottomRightCorner = (maxXCordinate, maxYCordinate)
        heightRegion =maxYCordinate - minYCordinate
        widthRegion = maxXCordinate - minXCordinate
        if (heightRegion >= heightMin and heightRegion <= heightMax):
            if(widthRegion >= widthMin and widthRegion <= widthMax):
                aspectRatio = float(widthRegion)/float(heightRegion)
                if (aspectRatio >= aspectRatioMin and aspectRatio <= aspectRatioMax):
                    rectCord = [leftTopCorner[0], leftTopCorner[1], widthRegion, heightRegion, aspectRatio]
                    bboxes.append(rectCord)
                    cv.rectangle(imageOrg, leftTopCorner, bottomRightCorner, (0, 0, 0), 2)
    numMserRegions = len(bboxes)
    maxCluster = np.minimum(numMserRegions, minNumClusters)
    print 'Num of MSER regions in Image - ', fName, ' - ',len(bboxes)
    bboxArray = np.array(bboxes)
#     for boxRun in xrange(numMserRegions):
#         curBox = bboxArray[boxRun,:]
#         leftTopCorner = (int(curBox[0]), int(curBox[1]))
#         imageArray[leftTopCorner[1]:leftTopCorner[1]+curBox[3],leftTopCorner[0]:leftTopCorner[0]+curBox[2]] = 255
#     plt.imshow(imageArray)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.show()
    if plotProgress:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(imageOrg)
        plt.show()
    # now cluster the data
    dataToCluster =  bboxArray[:,2:4]
    selectScales = getCenteroidsByGapStats(dataToCluster, maxCluster, plotProgress)
    return(selectScales)

def calculateWk(centroids, labels, dataToCluster):
    numOfClusters = np.max(labels) + 1
    wk = 0
    for clusterRun in xrange(numOfClusters):
        currentCenteroid = centroids[clusterRun,:]
        clusterPoints = dataToCluster[labels==clusterRun,:]
        numPointsInCluster = clusterPoints.shape[0]
        wk = wk + (np.linalg.norm(clusterPoints-currentCenteroid)**2 / float(2*numPointsInCluster))
    return wk

def getCenteroidsByGapStats(dataToCluster, maxCluster, plotProgress):
    # plot data
    if plotProgress:
        plt.scatter(dataToCluster[:,0],dataToCluster[:,1], s=100)
        #plt.xlabel('Width of MSER Bounding box (pixels)')
        #plt.ylabel('Height of MSER Bounding box (pixels)')
        plt.show()
    xminData = np.min(dataToCluster[:,0])
    xmaxData = np.max(dataToCluster[:,0])
    yminData = np.min(dataToCluster[:,1])
    ymaxData = np.max(dataToCluster[:,1])
    numOfClusterRuns = maxCluster - 1
    sumSqMetricSave = np.zeros((1, numOfClusterRuns))
    wksMetricSave = np.zeros((1, numOfClusterRuns))
    wkbsMetricSave = np.zeros((1, numOfClusterRuns))
    kMetricSave = np.zeros((1, numOfClusterRuns), dtype=np.int32)
    skMetricSave = np.zeros((1, numOfClusterRuns))
    centeroidSave = []
    labelSave = []
    for clusterRun in  xrange(1, maxCluster):
        centroids, labels, inertia = k_means(dataToCluster, n_clusters = clusterRun)
        centeroidSave.append(centroids)
        labelSave.append(labels)
        kMetricSave[0, clusterRun-1] = clusterRun
        # calculate gap stattistics for selecting the number of clusters
        tempVar = calculateWk(centroids, labels, dataToCluster)
        sumSqMetricSave[0, clusterRun-1] = tempVar
        wksMetricSave[0, clusterRun-1] = np.log(tempVar)
        # ref data set 
        bRef = 10
        BWkbs = np.zeros((1, bRef))
        for iRun in xrange(bRef):
            refData = np.zeros_like(dataToCluster) 
            for dataRun in xrange(dataToCluster.shape[0]):
                refData[dataRun,:] = np.array([np.random.uniform(xminData ,xmaxData), np.random.uniform(yminData, ymaxData)])
            centroidsRef, labelsRef, inertiaRef = k_means(refData, n_clusters = clusterRun)
            BWkbs[0, iRun] = np.log(calculateWk(centroidsRef, labelsRef, refData))
        wkbsMetricSave[0, clusterRun-1] = np.sum(BWkbs)/float(bRef)
        skMetricSave[0, clusterRun-1] = np.sqrt(np.sum((BWkbs - wkbsMetricSave[0, clusterRun-1])**2)/float(bRef))
    skMetricSave = skMetricSave*np.sqrt(1 + 1/float(bRef))
    # gap statistics
    gap = (wkbsMetricSave - wksMetricSave)
    gap= gap.reshape(1, -1)
    finalMetric = np.zeros((1, numOfClusterRuns))
    for iRun in xrange(1, maxCluster-1):
        # gapofk-gapofk+1-sk
        finalMetric[0, iRun-1] = gap[0, iRun-1] - (gap[0, iRun] - skMetricSave[0, iRun])
    
    if plotProgress:
        # 1
        plt.plot(np.arange(1,maxCluster), sumSqMetricSave.T,marker='s',markersize=10)
        #plt.xlabel('Number of clusters')
        #plt.ylabel('Height of MSER Bounding box (pixels)')
        plt.show()
        # 2
        plt.plot(np.arange(1,maxCluster), wksMetricSave.T,marker='o',markersize=10)
        plt.plot(np.arange(1,maxCluster), wkbsMetricSave.T,marker='D',markersize=10)
        #plt.xlabel('Number of clusters')
        #plt.ylabel('Height of MSER Bounding box (pixels)')
        plt.show()
        # 3
        plt.plot(np.arange(1,maxCluster), gap.T,marker='*',markersize=10)
        plt.show()
        # 4
        plt.bar(np.arange(1,maxCluster), finalMetric.T)
        #plt.xlabel('Number of clusters')
        #plt.ylabel('Height of MSER Bounding box (pixels)')
        plt.show()

    indeNonZero = np.where(finalMetric>0)[1]
    selectIndex = np.min(indeNonZero)
    # final clustering pics
    selectCenteroids =  np.array(centeroidSave[selectIndex])
    selectLabels = np.array(labelSave[selectIndex])
    if plotProgress:
        plt.scatter(dataToCluster[:,0],dataToCluster[:,1], c=selectLabels,s=100)
        plt.scatter(selectCenteroids[:,0],selectCenteroids[:,1], s=200, \
                    c= np.arange(selectIndex+1).astype(np.float32),marker='s')
        #plt.xlabel('Width of MSER Bounding box (pixels)')
        #plt.ylabel('Height of MSER Bounding box (pixels)')
        plt.show()
    return selectCenteroids

def detectActualScales(taggedRectangles, imagePath): 
    bboxes = []
    #imageArray = cv.imread(imagePath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    for eachTagRect in taggedRectangles.getchildren(): 
        word = eachTagRect.getchildren()[0].text
        segDetails = eachTagRect.getchildren()[1].getchildren()    
        eachTagRectDetails = eachTagRect.attrib
        coltag = int(float(eachTagRectDetails['x']))
        rowTag = int(float(eachTagRectDetails['y']))
        widthTag = int(float(eachTagRectDetails['width']))
        heightTag = int(float(eachTagRectDetails['height']))
        # calculate the char width
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
            aspectRatio =  float(widthCharTag)/float(heightCharTag)
            bboxes.append([colCharTag, rowCharTag, widthCharTag, heightCharTag, aspectRatio])
    return bboxes

def drawMSERProgression(imagePath,delta):
    imageArray = cv.imread(imagePath, cv.CV_LOAD_IMAGE_GRAYSCALE)   
    for thresholdRun in np.arange(0,255,delta):
        imagBin = imageArray < thresholdRun
        plt.imshow(imagBin, cmap = cm.get_cmap('Greys_r'))
        plt.savefig('threshold_'+str(thresholdRun)+'.png')