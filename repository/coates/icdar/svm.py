import numpy as np
from scipy import optimize

def myl2SVMlossCostFunction(weightMat, inputX, outputY , C, numOfLabels):
    # 1-vs-all L2-svm loss function;  similar to LibLinear.
    # based on adam coates
    #print 'C - ', C , ' lambda - ', 1/C
    # [M,N] = size(X);
    numOfExamples, numOfFeatures = inputX.shape
    # theta = reshape(w, N,K);
    theta = np.reshape(weightMat,(numOfFeatures,numOfLabels))
    #Y = bsxfun(@(y,ypos) 2*(y==ypos)-1, y, 1:K);
    yArray = np.zeros((numOfExamples, numOfLabels))
    for labelRun in np.arange(1, numOfLabels+1):
        yArray[:,labelRun-1] = (2*(outputY == labelRun)) - 1
    #margin = max(0, 1 - Y .* (X*theta));
    margin = np.maximum(0, 1 - np.multiply(yArray, np.dot(inputX, theta)))
    # loss = (0.5 * sum(theta.^2)) + C*mean(margin.^2);
    loss = (0.5 * np.sum(theta**2, 0)) + (C*np.mean(margin**2, 0))
    # loss = sum(loss);
    loss = np.sum(loss)   
    #g = theta - 2*C/M * (X' * (margin .* Y));
    svmgrad = theta - (2*C/float(numOfExamples)) * ((inputX.T).dot(np.multiply(margin, yArray)))
    # % adjust for intercept term
    #loss = loss - 0.5 * sum(theta(end,:).^2);
    loss = loss - 0.5 * np.sum(theta[-1,:]**2)
    #g(end,:) = g(end, :) - theta(end,:);
    svmgrad[-1,:] = svmgrad[-1, :] - theta[-1,:]
    svmgrad = svmgrad.T.flatten()
    return loss, svmgrad

def trainSVM(trainX, trainY, C, maxIterations=1000, numClasses=2):
    #w0 = zeros(size(trainXC,2)*numClasses, 1);
    w0 = np.zeros((trainX.shape[1]*numClasses, 1))
    myargs = (trainX, trainY, C, numClasses)
    # w = minFunc(@my_l2svmloss, w0, struct('MaxIter', 1000, 'MaxFunEvals', 1000), ...
    #          trainXC, trainY, numClasses, C);
    theta, cost, info = optimize.fmin_l_bfgs_b(myl2SVMlossCostFunction, w0, fprime = None, args=myargs, maxfun = 1000, disp=True)
    
    theta = np.reshape(theta, (trainX.shape[1], numClasses))
    return theta

def predictSVM(weightMat, featureArray):
    #[val,labels] = max(trainXCs*theta, [], 2);
    values = np.dot(featureArray,weightMat)
    labels = values.argmax()
    return values, labels

def addIntercept(trainFeatures):
    # Add intercept term (a column of ones) to x along vertical axis
    trainFeatures = np.hstack((trainFeatures, np.ones((trainFeatures.shape[0],1)))) 
    return trainFeatures

def score(trainX, trainY, theta):
    labels = predictSVM(theta, trainX)[1]
    accuracy = 1 - np.mean(labels != trainY)
    return accuracy*100
# svmModelPath = os.path.join(coates.resultPath, 'SVMModelSave.npy')
# if(os.path.isfile(svmModelPath)):
#     print 'SVM Model already generated'
#     print 'Loading the SVM Model - ', svmModelPath
#     pixelTextProbPredictorWeights = np.load(svmModelPath)
#     svmStats = np.load(trainingParamsPath)['svmStats']
#     print 'Selected Model C value - ', svmStats[0]
#     print 'Training accuracy - ', svmStats[1]
#     print 'Testing accuracy - ', svmStats[2]
# else:
#     # grid search
#     print 'Training the SVM'
#     # shuffle
#     dataIndex = np.arange(trainDataSet.shape[0])
#     np.random.shuffle(dataIndex)
#     np.random.shuffle(dataIndex)
#     trainDataSet = trainDataSet[dataIndex,:]
#     dataIndex = np.arange(testDataSet.shape[0])
#     np.random.shuffle(dataIndex)
#     np.random.shuffle(dataIndex)
#     testDataSet = testDataSet[dataIndex,:]
#     # lambda and C values are inversly proportional
#     lambdaVals = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
#     #cVals = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
#     modelCollect = []
#     precisionCollect = []
#     maxIterationsSVM=1000
#     numClasses=2
#     for curLambdaVal in lambdaVals:
#         C = 1/curLambdaVal
#         pixelTextProbPredictorWeights = svm.trainSVM(trainDataSet[:,1:], trainDataSet[:,0], C, maxIterationsSVM, numClasses)
#         # testing data
#         score = 0
#         for testRun in np.arange(3):
#             randomSelect = np.random.randint(0, testDataSet.shape[0], 8000)
#             score += svm.score(testDataSet[randomSelect,1:], testDataSet[randomSelect,0], pixelTextProbPredictorWeights)
#         print 'C value - ', C, ' Score - ', score/3.0 
#         precisionCollect.append(score/3.0)
#         modelCollect.append(pixelTextProbPredictorWeights)            
#     selectIndex = precisionCollect.index(max(precisionCollect))
#     pixelTextProbPredictorWeights =  modelCollect[selectIndex] 
#     selectedCval = 1/lambdaVals[selectIndex]
#     trainAccuracy = svm.score(trainDataSet[:,1:], trainDataSet[:,0], pixelTextProbPredictorWeights)
#     testAccuracy = svm.score(testDataSet[:,1:], testDataSet[:,0], pixelTextProbPredictorWeights)
#     svmStats = np.array([selectedCval, trainAccuracy, testAccuracy, precisionCollect[selectIndex]])
#     print 'Selected C value - ', selectedCval
#     print 'Model Precision - ', precisionCollect[selectIndex]
#     print 'Training accuracy - ', trainAccuracy
#     print 'Testing accuracy - ', testAccuracy
#     # save the model for prediction
#     np.save(pixelTextProbPredictorWeights, svmModelPath) 
#     np.savez_compressed(trainingParamsPath, meanAvgZCA = trainingParamsList[0],\
#                     whitenTransform = trainingParamsList[1], meanAvgValFeatures = feStandardized[0],
#                     stdAvgValFeatures = feStandardized[1], svmStats = svmStats)
#     
#     # memory clean up 
#     del modelCollect
#     del precisionCollect
#     del errorCollect 
# del testDataSet
# del trainDataSet
# #time.sleep(60)
