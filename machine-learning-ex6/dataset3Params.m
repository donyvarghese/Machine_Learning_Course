function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;
C_in=[0.01 0.03 0.1 0.3 1 3 10 30];
sigma_in=[0.01 0.03 0.1 0.3 1 3 10 30];
%C_in=[0.01 0.03];
%sigma_in=[0.01 0.03];
l_C = length(C_in);
l_sigma = length(sigma_in);
error = zeros(l_C,l_sigma);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i=1:l_C
	for j =1:l_sigma
		model= svmTrain(X, y, C_in(i), @(x1, x2) gaussianKernel(x1, x2, sigma_in(j)));
		predictions = svmPredict(model, Xval);
		error(i,j) = mean(double(predictions ~= yval));
	end
end
error;
[minval,ind] = min(error(:));
[C_index,sigma_index] = ind2sub([size(error,1) size(error,2)],ind);
%[C_index, sigma_index] = find(error == min(error(:)), 1)

C = C_in(C_index);
sigma = sigma_in(sigma_index);
%error
 


% =========================================================================

end
