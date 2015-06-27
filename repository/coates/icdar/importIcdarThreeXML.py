import numpy as np
import xml.etree.ElementTree as xmlParser
import os
import time
import cPickle as pickle
from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2 as cv
# helper  function
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
        wordDetails = [coltag, rowTag, coltag+widthTag, rowTag+heightTag]
        boundingBoxWord.append(wordDetails)
        charWidthArray = np.array([0])
        for tempRun in np.arange(len(segDetails)):
            charWidthArray = np.append(charWidthArray, int(float(segDetails[tempRun].text)))
        charWidthArray = np.append(charWidthArray, widthTag)
        charWidthArray = np.diff(charWidthArray)
        # run through the all the patches
        for eachCharRun in np.arange(len(word)):
            #curChar = word[eachCharRun]
            rowCharTag = rowTag
            heightCharTag = heightTag
            widthCharTag = charWidthArray[eachCharRun]
            if eachCharRun == 0:
                colCharTag = coltag
            else:
                colCharTag += charWidthArray[eachCharRun-1]
            # left top bottom right
            charDetails = [colCharTag, rowCharTag, colCharTag+widthCharTag, rowCharTag+heightCharTag]
            boundingBoxChar.append(charDetails)
    print 'Number of characters in image - ', len(boundingBoxChar)
    print 'Number of words in image - ', len(boundingBoxWord)
    bboxCharArray = np.array(boundingBoxChar)
    bboxWordArray = np.array(boundingBoxWord)
    return [bboxCharArray, bboxWordArray]
###################################################################################################
# script begin
dataBasePath = os.path.join('/home','rthalapp','masterThesis','datasets','icdar','test')
gTruthFolder = os.path.join(dataBasePath,'groundTruth')
charBboxInfosFolder = os.path.join(dataBasePath,'charBboxInfos')
imagePathSaveList = os.path.join(dataBasePath, 'imgPathList.pkl')
if not os.path.exists(charBboxInfosFolder):
    os.makedirs(charBboxInfosFolder)
if not os.path.exists(gTruthFolder):
    os.makedirs(gTruthFolder)
xmlName = 'segmentation.xml'
xmlCompletePath = os.path.join(dataBasePath, xmlName)
print 'Reading words Xml from ', xmlCompletePath
tree = xmlParser.parse(xmlCompletePath)
root = tree.getroot()
# get number of images listed in the XML
numOfImages = root.__len__()
imageRun = 1
imgPathList = []
allbboxesActual = []
# loop through each image 
###################################################################################################
for imageDetails in list(root):
    print '************************************************************************************'
    start = time.time()
    # get the image name
    # image name
    imgName = imageDetails.getchildren()[0].text
    # tagged rectangles
    tagRectangles = imageDetails.getchildren()[2] 
    imgPath = os.path.join(dataBasePath, imgName)
    imageOrg = cv.imread(imgPath)
    b,g,r = cv.split(imageOrg)
    imageOrg = cv.merge([r,g,b])
    # save the path
    imgPathList.append(imgPath)
    imageArray = np.round(imread(imgPath, flatten=True)).astype(np.uint8)
    gTruthArray = np.zeros_like(imageArray)
    dirName, fName = os.path.split(imgPath)
    fName = fName.split('.')[0]
    dirName = dirName.split('/')[-1]
    gTruthFileName = os.path.join(gTruthFolder, 'GTruth_' + dirName + '_' + fName + '.npy')
    charInfoFileName = os.path.join(charBboxInfosFolder, 'CharInfos_' + dirName + '_' + fName + '.npy')
    print imageRun, '. Image - ', imgPath , '(' , imageRun, '/', numOfImages,')'
    charDetailsArray, wordDetailsArray = getAllCharBoxesImage(tagRectangles)
    for charRun in xrange(charDetailsArray.shape[0]):
        curCharDetails = charDetailsArray[charRun,:]
        # x,y x -column y -rows
        leftTopCornerChar = (curCharDetails[0], curCharDetails[1])
        rightBottomCornerChar = (curCharDetails[2], curCharDetails[3])
        widthChar = rightBottomCornerChar[0] - leftTopCornerChar[0]
        heightChar = rightBottomCornerChar[1] - leftTopCornerChar[1]
        aspectRatio = float(widthChar)/float(heightChar)
        bboxesActual = [widthChar, heightChar, aspectRatio]
        allbboxesActual.append(bboxesActual)
        gTruthArray[leftTopCornerChar[1]:leftTopCornerChar[1]+heightChar, \
                    leftTopCornerChar[0]:leftTopCornerChar[0]+widthChar] = 1
        cv.rectangle(imageOrg, leftTopCornerChar, rightBottomCornerChar, (0, 0, 0), 2)
    # save the gTruth
    np.save(gTruthFileName, gTruthArray)
    np.save(charInfoFileName, charDetailsArray)
#     plt.imshow(imageOrg)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.show()
    end = time.time()
    print imageRun, '. Time elapsed - ', end-start
    imageRun = imageRun + 1
print '############################################################################################'
allbboxesActualArray = np.array(allbboxesActual) 
charArea = allbboxesActualArray[:,0].astype(np.float64)*allbboxesActualArray[:,1].astype(np.float64)
print 'Max aspect ratio - ', np.max(allbboxesActualArray[:,2])
print 'Min aspect ratio - ', np.min(allbboxesActualArray[:,2]) 
print 'Mean aspect ratio - ', np.mean(allbboxesActualArray[:,2]), ' ', np.std(allbboxesActualArray[:,2])
print 'Max area - ', np.max(charArea)
print 'Min area - ', np.min(charArea)
print 'Max height - ', np.max(allbboxesActualArray[:,1])
print 'Min height - ', np.min(allbboxesActualArray[:,1])
print 'Max width - ', np.max(allbboxesActualArray[:,0])
print 'Min width - ', np.min(allbboxesActualArray[:,0])
print '############################################################################################'
# save the lsi of image paths
with open(imagePathSaveList, 'wb') as pklFile:
    pickle.dump(imgPathList, pklFile)
    pklFile.close()
newList = []
eachNum = int(np.ceil(len(imgPathList)/26) + 1)
irun = 1
allNum = np.arange(len(imgPathList))
np.random.shuffle(allNum)
np.random.shuffle(allNum)
for eachStart in xrange(0,len(imgPathList),eachNum):
    partImagePathList = os.path.join(dataBasePath, 'segmentation_'+ str(irun) +'.pkl')
    newList = [imgPathList[eachIndex] for eachIndex in allNum[eachStart:eachStart+eachNum]]
    print len(newList)
    with open(partImagePathList, 'wb') as pklFile:
        pickle.dump(newList, pklFile)
        pklFile.close()
    irun += 1
print '############################################################################################'