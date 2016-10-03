#! /bin/octave -qf
function [all_theta] = oneVsAll(X, y, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);
global fmincgIter;
global num_labels;

% the following variable will be returned 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% The following code trains num_labels logistic regression classifiers 
% with regularization parameter lambda. 
%
%
% We can use y == c to obtain a vector of 1's and 0's with 1 if y==c
%
%     % Running fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(X, (y == c), t, lambda)), ...
%                 initial_theta, options);
%

for c=1:num_labels
	options = optimset('GradObj', 'on', 'MaxIter', fmincgIter);
	initial_theta = zeros(n + 1, 1);
    theta = fmincg (@(t)(lrCostFunction(X, y == c, t, lambda)), initial_theta, options);
    if (c==1)
    	all_theta = theta;
    else
        all_theta(c,:) = theta';
    endif


% =========================================================================


end
