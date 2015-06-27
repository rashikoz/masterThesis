#import vlfeat
import xml.etree.ElementTree as xmlParser
import coates
import os
import time
import mserHelper

xmlName = 'segmentation.xml'
xmlCompletePath = os.path.join(coates.trainPath, xmlName)
print 'Generating the Training Data'
print 'Reading words Xml from ', xmlCompletePath
tree = xmlParser.parse(xmlCompletePath)
root = tree.getroot()
# get number of images listed in the XML
numOfImages = root.__len__()
imageRun = 1
delta = 5
minArea = 30
maxArea = 100000
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
    # train examples
    #imgName = 'lfsosa_12.08.2002/IMG_2471.JPG'
    #imgName = 'ryoungt_05.08.2002/aPICT0006.JPG'
    # test examples
    imgName = 'ryoungt_05.08.2002/PICT0015.JPG'
    #imgName = 'ryoungt_05.08.2002/Pict0003.jpg'
    imgPath = os.path.join(coates.testPath, imgName)
    dirName, fName = os.path.split(imgPath)   
    fName = fName.split('.')[0]
    dirName = dirName.split('/')[-1]    
    print imageRun, '. Image - ', imgPath , '(' , imageRun, '/', numOfImages,')'
    #bboxesActual = mserHelper.detectActualScales(tagRectangles, imgPath)
    #allbboxesActual.extend(bboxesActual)
    #mserHelper.drawMSERProgression(imgPath, delta)
    bboxesDetected = mserHelper.detectMSERBboxes(imgPath, delta, minArea, maxArea, maxVariation, minDiversity, 1)
    imageRun += 1
# allbboxesActualArray = np.array(allbboxesActual) 
# charArea = allbboxesActualArray[:,2].astype(np.float64)*allbboxesActualArray[:,3].astype(np.float64)
# print 'Max aspect ratio - ', np.max(allbboxesActualArray[:,4])
# print 'Min aspect ratio - ', np.min(allbboxesActualArray[:,4]) 
# print 'Mean aspect ratio - ', np.mean(allbboxesActualArray[:,4]), ' ', np.std(allbboxesActualArray[:,4])
# print 'Max area - ', np.max(charArea)
# print 'Min area - ', np.min(charArea)


# xmlName = 'char.xml'
# charPath = os.path.join('/home','rashik','Downloads','dataset','train','char')
# xmlCompletePath = os.path.join(charPath, xmlName)
# print 'Reading words Xml from ', xmlCompletePath
# tree = xmlParser.parse(xmlCompletePath)
# root = tree.getroot()
# # get number of images listed in the XML
# numOfImages = root.__len__()
# imageRun = 1
# delta = 10
# minArea = 30
# maxArea = 100000
# maxVariation = 0.1
# minDiversity = 0.9
# allbboxesActual = []
# for imageDetails in list(root):
#     tag = imageDetails.attrib['tag']
#     fName = imageDetails.attrib['file']
#     imgPath = os.path.join(charPath, fName)
#     bboxesDetected = mserHelper.detectMSERBboxes(imgPath, delta, minArea, maxArea, maxVariation, minDiversity)

