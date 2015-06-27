import numpy as np
import sys
import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
import sklearn.metrics as metrics 
from scipy.misc import imread
import coates
import helper
####################################################################################################
# 0- file name
# 1 - number of clusters
if(len(sys.argv) > 1):
    coates.numOfClusters = int(sys.argv[1])
print 'Number of clusters - ',coates.numOfClusters
coates.resultPath = os.path.join(coates.resultPath,str(coates.numOfClusters))
####################################################################################################
# Now analysis  
testFolder = coates.testPath
resultsFolder = coates.resultPath
gTruthFolder = os.path.join(coates.testPath, 'groundTruth')
predictFolder = os.path.join(resultsFolder, 'prediction')
maxPredictFolder = os.path.join(predictFolder, 'maxPredict')
# result files
uniqueThresholdResults = os.path.join(resultsFolder, 'uniqueThresholds.npz')
fixedThresholdResults = os.path.join(resultsFolder, 'fixedThresholds.npz')
checkPresence = os.path.isfile(uniqueThresholdResults) and \
                os.path.isfile(fixedThresholdResults)
imageRun = 1
numOfthresholds = 1000
minT  = 0
maxT = 0
###################################################################################################
# calculate max image
maxPredictFileList = os.listdir(maxPredictFolder)
numMaxFiles = len(maxPredictFileList) 
print 'Reading folder - ', maxPredictFolder
print 'Number of files in directory - ',  numMaxFiles
for eachMaxPredictFile in maxPredictFileList:
    if eachMaxPredictFile.endswith(".npy"):
        if eachMaxPredictFile.startswith('MaxPredict_'):
            maxPredict = np.load(os.path.join(maxPredictFolder, eachMaxPredictFile))
            minT = np.minimum(minT, np.min(maxPredict))
            maxT = np.maximum(maxT, np.max(maxPredict))
print 'Minimum Threshold value  - ', minT
print 'Maximum Threshold value  - ', maxT
###################################################################################################
# PR calculation
if(checkPresence):
    print 'Data already present.Data loading from - ', resultsFolder
    # unique thresholds
    avPrecisionArray = np.load(uniqueThresholdResults)['avPrecisionArray']
    precisionArray = np.load(uniqueThresholdResults)['precisionArray']
    recallArray = np.load(uniqueThresholdResults)['recallArray']
    tpsArray = np.load(uniqueThresholdResults)['tpsArray']
    fpsArray = np.load(uniqueThresholdResults)['fpsArray']
    fnsArray = np.load(uniqueThresholdResults)['fnsArray']
    thresholdLevels = np.load(uniqueThresholdResults)['thresholdLevels']    
    # fixed threshold values
    truePositiveArrayThresholded = np.load(fixedThresholdResults)['truePositiveArrayThresholded']
    falsePositiveArrayThresholded = np.load(fixedThresholdResults)['falsePositiveArrayThresholded']
    falseNegativeArrayThresholded = np.load(fixedThresholdResults)['falseNegativeArrayThresholded']
    avPrecisionArrayThresholded = np.load(fixedThresholdResults)['avPrecisionArrayThresholded']
    precisionArrayThresholded = np.load(fixedThresholdResults)['precisionArrayThresholded']
    recallArrayThresholded = np.load(fixedThresholdResults)['recallArrayThresholded']    
else:
    # fixed threshold values
    truePositiveArrayThresholded = np.zeros((numMaxFiles, numOfthresholds))
    falsePositiveArrayThresholded = np.zeros((numMaxFiles, numOfthresholds))
    falseNegativeArrayThresholded = np.zeros((numMaxFiles, numOfthresholds))
    avPrecisionArrayThresholded = np.asarray([], dtype = np.float32)
    precisionArrayThresholded = np.asarray([], dtype = np.float32)
    recallArrayThresholded = np.asarray([], dtype = np.float32)
    # unique thresholds
    avPrecisionArray = np.asarray([], dtype = np.float32)
    precisionArray = np.asarray([], dtype = np.float32)
    recallArray = np.asarray([], dtype = np.float32)
    tpsArray = np.asarray([])
    fpsArray = np.asarray([])
    fnsArray = np.asarray([])
    thresholdLevels = np.asarray([], dtype = np.float32)
    for eachMaxPredictFile in maxPredictFileList:
        if eachMaxPredictFile.endswith(".npy"):
            if eachMaxPredictFile.startswith('MaxPredict_'):
                print imageRun, '. Loading - ',eachMaxPredictFile
                # get the file name
                fName = eachMaxPredictFile.partition('MaxPredict_')[2]
                picName = fName.split('_')[2]
                folderName = fName.partition('_'+picName)[0]
                picName = picName.replace('.npy', '.jpg')
                picFileFullpath = os.path.join(testFolder,folderName,picName)
                if not (os.path.isfile(picFileFullpath)):
                    picFileFullpath = picFileFullpath.replace('.jpg', '.JPG')
                # get the corresponding ground truth
                gTruthFilename = 'GTruth_' + fName
                print imageRun, '. Loading - ',gTruthFilename
                gTruthArray = np.load(os.path.join(gTruthFolder, gTruthFilename))
                maxPredict = np.load(os.path.join(maxPredictFolder, eachMaxPredictFile))
                orgImageArray = imread(picFileFullpath)
                showMaxPredict = maxPredict.copy()
                showMaxPredict[showMaxPredict < 0] = 0
                plt.imshow(orgImageArray)
                plt.imshow(showMaxPredict, cmap = cm.get_cmap('Greys_r'), alpha = 0.8)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.show()
                gTruthFlat = gTruthArray.ravel()
                predictFlat = maxPredict.ravel()
                # increased value order
                incIndices = np.argsort(predictFlat, kind="mergesort")
                predictFlat = predictFlat[incIndices]
                gTruthFlat = gTruthFlat[incIndices]
                # min and max values
                #minT = np.minimum(minT, predictFlat[0])
                #maxT = np.maximum(maxT, predictFlat[-1])
                # we calculate the fps,tps and fns values at fixed thresholds
                # fixed thresholds 
                tpsVals, fpsVals, fnsVals, precisionVals, recallVals, thresholdvals = \
                                                    helper.prCalculatorFast(predictFlat, gTruthFlat,\
                                                                            1, numOfthresholds, minT, maxT)
                truePositiveArrayThresholded[imageRun-1, :] = tpsVals
                falsePositiveArrayThresholded[imageRun-1, :] = fpsVals
                falseNegativeArrayThresholded[imageRun-1, :] = fnsVals 
                incIndices = np.argsort(recallVals, kind="mergesort")
                recallVals = recallVals[incIndices]
                precisionVals = precisionVals[incIndices] 
                avPrecision =  metrics.auc(np.append(0, recallVals), np.append(1, precisionVals) )
                print '----Average Precision fixed thresholds - ', avPrecision
                avPrecisionArrayThresholded = np.append(avPrecisionArrayThresholded, avPrecision)
                precisionArrayThresholded = np.append(precisionArrayThresholded, precisionVals)
                recallArrayThresholded = np.append(recallArrayThresholded,recallVals)
                if imageRun < 50:
                    if imageRun == 1:
                        fig1 = plt.figure(1)
                        axThresholded = fig1.add_subplot(111)
                    # give the start point on the xaxis                    
                    axThresholded.plot(np.append(0, recallVals),np.append(1, precisionVals))
                # unique thresholds evaluation    
                # the following code calculates pr values at unique thresholds
                # also logs tps,fps and fns values
                tpsVals, fpsVals, fnsVals, precisionVals, recallVals, thresholdvals = \
                                                     helper.prCalculatorFast(predictFlat, gTruthFlat)
                tpsArray = np.append(tpsArray,tpsVals)
                fpsArray = np.append(fpsArray,fpsVals)
                fnsArray = np.append(fnsArray,fnsVals)
                thresholdLevels = np.append(thresholdLevels, thresholdvals)
                avPrecision = metrics.auc(recallVals, precisionVals)
                print '----Average Precision Unique thresholds - ', avPrecision
                # save the PR and av precision
                precisionArray = np.append(precisionArray, precisionVals)
                recallArray = np.append(recallArray,recallVals)
                avPrecisionArray = np.append(avPrecisionArray, avPrecision)
                # plot the overlay figures
                if imageRun < 50:
                    if imageRun == 1:
                        fig2 = plt.figure(2)
                        axUnique = fig2.add_subplot(111)
                    # give the start point on the xaxis
                    axUnique.plot(recallVals,precisionVals)
                imageRun += 1
    # save the files
    np.savez(uniqueThresholdResults, avPrecisionArray=avPrecisionArray,\
              precisionArray=precisionArray, recallArray=recallArray,\
              tpsArray=tpsArray, fpsArray=fpsArray, fnsArray=fnsArray,\
              thresholdLevels=thresholdLevels)
    # fixed threshold values
    np.savez(fixedThresholdResults, truePositiveArrayThresholded=truePositiveArrayThresholded,\
             falsePositiveArrayThresholded=falsePositiveArrayThresholded,\
             falseNegativeArrayThresholded=falseNegativeArrayThresholded,\
             avPrecisionArrayThresholded=avPrecisionArrayThresholded,\
             precisionArrayThresholded=precisionArrayThresholded,\
             recallArrayThresholded=recallArrayThresholded)        
    # save the overlayed pr curves
    # fixed thresholds
    plt.figure(1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.00])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curves of 50 images Fixed thresholds')
    plt.savefig(os.path.join(resultsFolder, 'superImposedPRCurvesFixed.png'))
    plt.close(fig1)    
    # unique thresholds    
    plt.figure(2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.00])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curves of 50 images Unique thresholds')
    plt.savefig(os.path.join(resultsFolder, 'superImposedPRCurvesUnique.png'))
    plt.close(fig2)
print '############################################################################################'
# now the calculation
print 'Fixed thresholds'
mAP = np.mean(avPrecisionArrayThresholded)
print 'Minimum Threshold value  - ', minT
print 'Maximum Threshold value  - ', maxT 
print 'Mean Average Precision (mAP) fixed thresholds - ', mAP
# calculate the precision and recall 
tpsSum = np.sum(truePositiveArrayThresholded, 0).reshape(1,-1)
fpsSum = np.sum(falsePositiveArrayThresholded, 0).reshape(1,-1)
fnsSum = np.sum(falseNegativeArrayThresholded, 0).reshape(1,-1)
finalPrecision =  tpsSum/(tpsSum + fpsSum)
finalRecall = tpsSum/(tpsSum + fnsSum)
nanIndex = np.isnan(finalPrecision)
finalPrecision[nanIndex] = 1
finalRecall[nanIndex] = 0
finalPrecision = np.append(finalPrecision, 1)
finalRecall = np.append(finalRecall, 0)
# calculate and plot the graph
incRecallIndices = np.argsort(finalRecall, kind="mergesort")
finalRecall = finalRecall[incRecallIndices]
finalPrecision = finalPrecision[incRecallIndices]
auc = metrics.auc(finalRecall, finalPrecision)
print 'Area under Precision Recall Curve fixed thresholds - ', auc  
plt.clf()
plt.plot(finalRecall, finalPrecision, label= str(coates.numOfClusters) + \
         ' Centeroids - AUC={0:0.3f}'.format(auc) + \
         ' mAP = {0:0.3f}'.format(mAP))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.00])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig(os.path.join(resultsFolder, 'prCurveFixedThresholds.png'))
print '############################################################################################'
print 'Unique thresholds'
mAP = np.mean(avPrecisionArray)
print 'Mean Average Precision (mAP) Unique thresholds - ', mAP
print '############################################################################################'
# CODE INCLUDED HERE so that we can c the results from the whole process in one file
svmModelPath = os.path.join(coates.resultPath, 'svmModel.mat')
testDataPath = os.path.join(coates.resultPath, 'testDataSet.npy')
trainDataPath = os.path.join(coates.resultPath, 'trainingSet.npy')
print 'Training Path - ', coates.trainPath
print 'Testing Path - ', coates.testPath
print 'Loading the SVM Model - ', svmModelPath
matContents = loadmat(svmModelPath)
svmStats = matContents['svmStats']
print 'Selected Model lambda value - ', svmStats[0, 0], ' C - ', 1/svmStats[0, 0]
print 'Training accuracy - ', svmStats[0, 1]
print 'Testing accuracy - ', svmStats[0, 2]
exit()
print '############################################################################################'
