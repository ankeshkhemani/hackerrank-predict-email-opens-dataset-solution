#! /usr/local/bin/octave -qf

function [error_train, error_val] = learningCurve(X, y, curve_type)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, curve_type) returns the train and
%       cross validation set errors for a chosen learning curve type. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%

global selected_poly;               % Used in curve_type 1,3
global lambda;               % Used in curve_type 1,2
global learning_curve_stabiliser;   % Used in curve_type 1,2,3
global learning_examples;  % Used in curve_type 1
global polynomial_degree;  % Used in curve_type 2
global lambda_vec;         % Used in curve_type 3
global test_sample_size;   %Used in 2,3
% Number of training examples
m = size(X, 1);

% Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%

if (curve_type == 1)     %Data Size Learning Curve
% You need to return these values correctly
error_train = zeros(learning_examples, 1);
error_val   = zeros(learning_examples, 1);

%Apply selected polynomial_degree
[X_poly] = polyFeatures(X, selected_poly);
[X_norm, mu, sigma] = featureNormalize(X_poly);
X_poly=X_norm;


for i = 1:learning_examples    %For each example on the learning Curve
	for j=1:learning_curve_stabiliser    	%Take average over several examples for each point

		%Randomize dataset
		n = rand(length(X_poly),1); 
        [garbage index] = sort(n); 
        Xrand = X_poly(index,:); 
        yrand = y(index,:);

        %Take first i+2 examples for training
        Xtrain = Xrand(1:i+2,:);
        ytrain = yrand(1:i+2,:);

        %Take next i+2 examples for validation
        Xval = Xrand(i+3:(2*i+4),:);
        yval =yrand(i+3:(2*i+4),:);

        [theta] = train(Xtrain, ytrain, lambda);
        train_error = calculateError(Xtrain, ytrain, theta);
        val_error = calculateError(Xval, yval, theta);
        
        error_train(i)=error_train(i)+train_error;
        error_val(i)=error_val(i)+val_error;
    endfor
    error_train(i) = error_train(i) / learning_curve_stabiliser;
    error_val(i) = error_val(i) / learning_curve_stabiliser;
endfor

elseif (curve_type == 2)     %Polynomial Learning Curve
% You need to return these values correctly
error_train = zeros(polynomial_degree, 1);
error_val   = zeros(polynomial_degree, 1);

for p = 1:polynomial_degree    %For each polynomial_degree on the learning Curve
    [X_poly] = polyFeatures(X, p);
    [X_norm, mu, sigma] = featureNormalize(X_poly);
    X_poly=X_norm;

    for j=1:learning_curve_stabiliser       %Take average over several examples for each point
        %Randomize dataset
        n = rand(length(X),1); 
        [garbage index] = sort(n); 
        Xrand = X_poly(index,:); 
        yrand = y(index,:);

        %Take first test_sample_size examples for training
        Xtrain = Xrand(1:test_sample_size,:);
        ytrain = yrand(1:test_sample_size,:);

        %Take next test_sample_size examples for validation
        Xval = Xrand(test_sample_size+1:2*test_sample_size,:);
        yval =yrand(test_sample_size+1:2*test_sample_size,:);

        [theta] = train(Xtrain, ytrain, lambda);
        train_error = calculateError(Xtrain, ytrain, theta);
        val_error = calculateError(Xval, yval, theta);
        
        error_train(p)=error_train(p)+train_error;
        error_val(p)=error_val(p)+val_error;
    endfor
    error_train(p) = error_train(p) / learning_curve_stabiliser;
    error_val(p) = error_val(p) / learning_curve_stabiliser;
endfor

elseif (curve_type == 3)     %Cross validation Learning Curve(for Selecting lambda)

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

%Apply selected polynomial_degree
[X_poly] = polyFeatures(X, selected_poly);
[X_norm, mu, sigma] = featureNormalize(X_poly);
X_poly=X_norm;

for l = 1:length(lambda_vec)          %For each lambda on the validation Curve
    trylambda = lambda_vec(l);
    for j=1:learning_curve_stabiliser       %Take average over several examples for each point
        %Randomize dataset
        n = rand(length(X_poly),1); 
        [garbage index] = sort(n); 
        Xrand = X_poly(index,:); 
        yrand = y(index,:);

        %Take first test_sample_size examples for training
        Xtrain = Xrand(1:test_sample_size,:);
        ytrain = yrand(1:test_sample_size,:);

        %Take next test_sample_size examples for validation
        Xval = Xrand(test_sample_size+1:2*test_sample_size,:);
        yval = yrand(test_sample_size+1:2*test_sample_size,:);

        [theta] = train(Xtrain, ytrain, trylambda);
        train_error = calculateError(Xtrain, ytrain, theta);
        val_error = calculateError(Xval, yval, theta);
        
        error_train(l)=error_train(l)+train_error;
        error_val(l)=error_val(l)+val_error;
    endfor
    error_train(l) = error_train(l) / learning_curve_stabiliser;
    error_val(l) = error_val(l) / learning_curve_stabiliser;
endfor
endif

% -------------------------------------------------------------

% =========================================================================

end
