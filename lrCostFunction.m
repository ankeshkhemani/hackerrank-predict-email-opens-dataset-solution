#! /bin/octave -qf
function [J, grad] = lrCostFunction(X, y, theta, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% number of training examples
m = length(y); 

% return the following variables  
J = 0;
grad = zeros(size(theta));

% Compute the cost of a particular choice of theta in J
% Compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta


predictions=sigmoid(X*theta);

%using sum
errors= - ((y .* log(predictions)) + ((1 - y) .* (log(1 - predictions))));
J=1/m * sum(errors) + lambda/(2 * m) * sum(theta(2:end).^ 2);


grad= (1/m) * X' * (sigmoid(X * theta) - y);
grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);

% =============================================================

grad = grad(:);

end
