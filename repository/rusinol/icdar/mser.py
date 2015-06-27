import numpy as np
import xml.etree.ElementTree as xmlParser
import rusinol
import os
import time
import mserHelper

xmlName = 'segmentation.xml'
xmlCompletePath = os.path.join(rusinol.trainPath, xmlName)
print 'Generating the Training Data'
print 'Reading words Xml from ', xmlCompletePath
tree = xmlParser.parse(xmlCompletePath)
root = tree.getroot()
# get number of images listed in the XML
numOfImages = root.__len__()
imageRun = 1
delta = 5
minArea = 30
maxArea = 90000
maxVariation = 0.2
minDiversity = 0.1
allbboxesActual = []
for imageDetails in list(root):
    print '************************************************************************************'
    start = time.time()
    # get the image name
    imgName = imageDetails.getchildren()[0].text
    # resolution
    resolution = imageDetails.getchildren()[1]
    # tagged rectangles
    tagRectangles = imageDetails.getchildren()[2]    
    imgPath = os.path.join(rusinol.trainPath, imgName)
    dirName, fName = os.path.split(imgPath)   
    fName = fName.split('.')[0]
    dirName = dirName.split('/')[-1]    
    print imageRun, '. Image - ', imgPath , '(' , imageRun, '/', numOfImages,')'
    bboxesActual = mserHelper.detectActualScales(tagRectangles, imgPath)
    allbboxesActual.extend(bboxesActual)
    bboxesDetected = mserHelper.detectMSERBboxes(imgPath, delta, minArea, maxArea, maxVariation, minDiversity)
    imageRun += 1
allbboxesActualArray = np.array(allbboxesActual) 
charArea = allbboxesActualArray[:,2].astype(np.float64)*allbboxesActualArray[:,3].astype(np.float64)
print 'Max aspect ratio - ', np.max(allbboxesActualArray[:,4])
print 'Min aspect ratio - ', np.min(allbboxesActualArray[:,4]) 
print 'Mean aspect ratio - ', np.mean(allbboxesActualArray[:,4]), ' ', np.std(allbboxesActualArray[:,4])
print 'Max area - ', np.max(charArea)
print 'Min area - ', np.min(charArea)
print 'Max height - ', np.max(allbboxesActualArray[:,3])
print 'Min height - ', np.min(allbboxesActualArray[:,3]) 
print 'Max width - ', np.max(allbboxesActualArray[:,2])
print 'Min width - ', np.min(allbboxesActualArray[:,2]) 
