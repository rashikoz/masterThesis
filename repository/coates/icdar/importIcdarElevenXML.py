import numpy as np
import os
import cPickle as pickle
from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2 as cv
# script begin
dataBasePath = os.path.join('/home','rthalapp','masterThesis','datasets','icdar11','train')
trainTrue = True
gTruthFolder = os.path.join(dataBasePath,'groundTruth')
charBboxInfosFolder = os.path.join(dataBasePath,'charBboxInfos')
imagePathSaveList = os.path.join(dataBasePath, 'imgPathList.pkl')
imageFolder = os.path.join(dataBasePath, 'images')
if not os.path.exists(charBboxInfosFolder):
    os.makedirs(charBboxInfosFolder)
if not os.path.exists(gTruthFolder):
    os.makedirs(gTruthFolder)
imgPathList = []
listFiles = os.listdir(imageFolder)
numofFiles = len(listFiles)
print 'Number of files in directory - ',  numofFiles
imageRun = 1
allbboxesActual = []
# loop through each image 
###################################################################################################
for imgFile in listFiles:
    if imgFile.endswith(".jpg"):
        print imageRun, '. Loading - ',imgFile
        fName = imgFile.partition('.jpg')[0]
        charFName = fName + '_GT.txt'
        wordFName = 'gt_' + fName + '.txt'
        imgPath = os.path.join(imageFolder, imgFile)
        imageOrg = cv.imread(imgPath)
        b,g,r = cv.split(imageOrg)
        imageOrg = cv.merge([r,g,b])
        imgPathList.append(imgPath)
        charLevelGTFile = os.path.join(dataBasePath, 'GT_CharLevel', charFName)
        imageArray = np.round(imread(imgPath, flatten=True)).astype(np.uint8)
        gTruthArray = np.zeros_like(imageArray)
        dirName = os.path.split(imgPath)[0]
        dirName = dirName.split('/')[-1]
        gTruthFileName = os.path.join(gTruthFolder, 'GTruth_' + dirName + '_' + fName + '.npy')
        if (trainTrue == True):
            charGtruthInfo = open(charLevelGTFile, "r")
            charDetailsList = charGtruthInfo.readlines()
            charInfoFileName = os.path.join(charBboxInfosFolder, 'CharInfos_' + dirName + '_' + fName + '.npy')
            boundingBoxChar = []
            for eachCharRun in xrange(len(charDetailsList)):
                curCharDetails = str(charDetailsList[eachCharRun])
                if (not curCharDetails.startswith('#')) and (not curCharDetails.isspace()):
                    eachCharDetails = curCharDetails.split()
                    leftTopCordinateChar = (int(eachCharDetails[5]),int(eachCharDetails[6]))
                    rightBottomCordinateChar = (int(eachCharDetails[7]),int(eachCharDetails[8]))
                    widthOfChar= rightBottomCordinateChar[0] - leftTopCordinateChar[0]
                    heightOfChar = rightBottomCordinateChar[1] - leftTopCordinateChar[1]
                    aspectRatio = float(widthOfChar)/float(heightOfChar)
                    bboxesActual = [widthOfChar, heightOfChar, aspectRatio]
                    allbboxesActual.append(bboxesActual)
                    cv.rectangle(imageOrg, leftTopCordinateChar, rightBottomCordinateChar, (0, 0, 0), 2)
                    gTruthArray[leftTopCordinateChar[1]:leftTopCordinateChar[1]+heightOfChar, \
                                leftTopCordinateChar[0]:leftTopCordinateChar[0]+widthOfChar] = 1
                    charDetails = [leftTopCordinateChar[0], leftTopCordinateChar[1], \
                                   rightBottomCordinateChar[0], rightBottomCordinateChar[1]]
                    boundingBoxChar.append(charDetails)
            np.save(charInfoFileName, np.array(boundingBoxChar))
            np.save(gTruthFileName, gTruthArray)
#             plt.imshow(imageOrg)
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.show()
        else:
            wordFName = 'gt_' + fName + '.txt'
            wordLevelGTFile = os.path.join(dataBasePath, 'GT_WordLevel', wordFName)
            wordGtruthInfo = open(wordLevelGTFile, "r")
            wordDetailsList = wordGtruthInfo.readlines()
            for wordRun in xrange(len(wordDetailsList)):
                curWord  = wordDetailsList[wordRun]
                eachWordDetails = curWord.split(',')
                theWord = eachWordDetails[4].split('"')[1]
                numChars = len(theWord)
                leftTopCordinate = (int(eachWordDetails[0]),int(eachWordDetails[1]))
                rightBottomCordinate = (int(eachWordDetails[2]),int(eachWordDetails[3]))
                widthOfWord = rightBottomCordinate[0] - leftTopCordinate[0]
                heightOfWord = rightBottomCordinate[1] - leftTopCordinate[1]
                gTruthArray[leftTopCordinate[1]:leftTopCordinate[1]+heightOfWord, \
                                leftTopCordinate[0]:leftTopCordinate[0]+widthOfWord] = 1
            np.save(gTruthFileName, gTruthArray)
#         plt.imshow(imageArray)
#         plt.show()
#         plt.imshow(gTruthArray)
#         plt.show()
        imageRun += 1
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
print 'Mean height - ', np.mean(allbboxesActualArray[:,1])
print 'StdDev height - ', np.std(allbboxesActualArray[:,1])
print 'Max width - ', np.max(allbboxesActualArray[:,0])
print 'Min width - ', np.min(allbboxesActualArray[:,0])
print 'Mean width - ', np.mean(allbboxesActualArray[:,0])
print 'StdDev width - ', np.std(allbboxesActualArray[:,0])
print '############################################################################################'
# save the lsi of image paths
with open(imagePathSaveList, 'wb') as pklFile:
    pickle.dump(imgPathList, pklFile)
    pklFile.close()
newList = []
eachNum = int(np.ceil(len(imgPathList)/20) + 1)
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