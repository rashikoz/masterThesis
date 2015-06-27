import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn

# sub plots
def displayImgs(imgs, layout, plotsize = 1.1):
    nrows, ncols = layout
    fig, axes = plt.subplots(nrows, ncols, figsize = (plotsize * ncols, plotsize * nrows))
    fig.subplots_adjust(hspace = 0, wspace = 0)
    axes = axes.flatten()
    for i, img in enumerate(imgs):
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        axes[i].imshow(img, interpolation = 'nearest', cmap = cm.get_cmap('Greys_r'))
        
def eigMatlab(inputArray):
    eigenValues,eigenVectors = np.linalg.eigh(inputArray)
    idx = eigenValues.argsort()[::-1]  
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return [eigenValues, eigenVectors]


def prCalculatorFast(yScore, yTrue, thresholdMethod = 0, numOfthresholds = 1000, minT = 0, maxT=0):
    # test implentation
    # 1 = TPR(S) + FNR(S),   1 = TNR(S) + FPR(S).
    # TPR = TP(S) / P,      FNR = FN(S) / P,
    # TNR = TN(S) / N,      FPR = FP(S) / N,
    # unique threshold implemenatation
    numtextPixels = (yTrue == 1).sum()
    numnonTextPixels = (yTrue == 0).sum()
    fpr = np.array([])
    tpr = np.array([])
    thresholds = np.array([])
    if thresholdMethod == 0:
        # unique thresholding
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(yTrue, yScore)
        # rest calcultion
        fnr = 1 - tpr
        fps = fpr*numnonTextPixels
        tps = tpr*numtextPixels
        fns = fnr*numtextPixels      
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(yTrue, yScore) 
    else:
        # fixed thresholding
        #yTemp = 1.0/(1+np.exp(-1*yScore))
        #for thresholdRun in np.linspace(0, 1.1, numOfthresholds):
        for thresholdRun in np.linspace(minT, maxT, numOfthresholds):
            #predictBin = yTemp > thresholdRun
            predictBin = yScore > thresholdRun            
            tpsVal = np.sum(np.logical_and(predictBin, yTrue))    
            fpsVal = np.sum(np.logical_and(predictBin, np.logical_not(yTrue))) 
            fprEach = fpsVal/float(numnonTextPixels)
            tprEach = tpsVal/float(numtextPixels)
            fpr = np.append(fpr, fprEach)
            tpr = np.append(tpr, tprEach)
            thresholds = np.append(thresholds, thresholdRun)
        # rest calculation
        fnr = 1 - tpr
        fps = fpr*numnonTextPixels
        tps = tpr*numtextPixels
        fns = fnr*numtextPixels
        # pr calculation
        precision = tps/(fps + tps).astype(np.float32)
        recall = tpr
        nanIndex = np.isnan(precision)
        precision[nanIndex] = 1
        recall[nanIndex] = 0
    return tps, fps, fns, precision, recall, thresholds
