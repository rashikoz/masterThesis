function theta = trainSvm(trainXC, trainY, C)
% setting few options for min func
options.Method = 'lbfgs';
options.maxIter = 1000;
options.display = 'on';
options.TolFun = 1e-5;
options.DerivativeCheck = 'off';
options.MaxFunEvals = 1500;

numClasses = max(trainY);
w0 = zeros(size(trainXC,2)*numClasses, 1);
% based on adams method
w = minFunc(@l2svmloss, w0, options, trainXC, trainY, numClasses, C);
theta = reshape(w, size(trainXC,2), numClasses);
end


% 1-vs-all L2-svm loss function;  similar to LibLinear.
function [loss, g] = l2svmloss(w, X, y, K, C)
    [M,N] = size(X);
    theta = reshape(w, N,K);
    Y = bsxfun(@(y,ypos) 2*(y==ypos)-1, y, 1:K);

    margin = max(0, 1 - Y .* (X*theta));
    loss = (0.5 * sum(theta.^2)) + C*mean(margin.^2);
    loss = sum(loss);
    g = theta - 2*C/M * (X' * (margin .* Y));

    % adjust for intercept term
    loss = loss - 0.5 * sum(theta(end,:).^2);
    g(end,:) = g(end, :) - theta(end,:);
    g = g(:);
end
