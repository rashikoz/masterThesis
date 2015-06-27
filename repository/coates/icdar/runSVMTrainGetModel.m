function runSVMTrainGetModel(resultFolder)
curPath = genpath('.');
addpath(curPath);
% load the test and train data
trainDataPath = fullfile(resultFolder,'trainingSetMatlab.mat');
testDataPath = fullfile(resultFolder,'testDataSetMatlab.mat');
load(trainDataPath);
load(testDataPath);
%% svm model creation
svmDataPath = fullfile(resultFolder,'svmModel.mat');
trainfeatures = trainData(:,2:end);
trainOutputs = trainData(:,1);
clear trainData
testfeatures = testData(:,2:end);
testOutputs = testData(:,1);
clear testData
if (exist(svmDataPath,'file') == 0)
    lambdaValues = [0.001,0.003,0.01,0.03,0.1,0.3];
    trainAccuracyCollect = [];
    testAccuracyCollect = [];
    modelCollect = {};
    disp('Creating the SVM Model')
    for lambdaRun  = 1:length(lambdaValues)
        lambda = lambdaValues(lambdaRun);
        theta = trainSvm(trainfeatures, trainOutputs, 1/lambda);
        modelCollect{lambdaRun} = theta;
        % train error
        [~, predLabels] = max(trainfeatures*theta, [], 2);
        trainErr = sum(predLabels ~= trainOutputs)/length(trainOutputs);
        trainAccuracy = (1 - trainErr)*100;
        trainAccuracyCollect = [trainAccuracyCollect, trainAccuracy];
        % test error
        testAccuracy = 0;
        for irun =1:5
            randomSelect = randi(size(testfeatures, 1), 40000, 1);
            testFeSelect = testfeatures(randomSelect,:);
            labelsFeSelect = testOutputs(randomSelect,:);
            [~, predLabels] = max(testFeSelect*theta, [], 2);
            testErr = sum(predLabels ~= labelsFeSelect)/length(labelsFeSelect);
            testAccuracy = testAccuracy + (1 - testErr)*100;
        end
        testAccuracy = testAccuracy/5;
        testAccuracyCollect = [testAccuracyCollect, testAccuracy];
        disp(['Training  for lambda - ', num2str(lambda) , ' C - ', num2str(1/lambda)])
        disp (['Testing accuracy - ', num2str(testAccuracy)]);
        disp (['Training accuracy - ', num2str(trainAccuracy)]);
        pause(5);
    end
    [maxVal, bestCVNum] = max(testAccuracyCollect);
    lambda = lambdaValues(bestCVNum);
    trainAccuracy = trainAccuracyCollect(bestCVNum);
    testAccuracy = testAccuracyCollect(bestCVNum);
    theta = modelCollect{bestCVNum};
    svmStats = [lambda, trainAccuracy, testAccuracy];
    disp(['Best lambda  - ', num2str(lambda),' C - ', num2str(1/lambda)]);
    disp (['Testing accuracy - ', num2str(testAccuracy)]);
    disp (['Training accuracy - ', num2str(trainAccuracy)]);
    save(svmDataPath,'theta', 'svmStats');
else
    load(svmDataPath)
end
exit;
end

