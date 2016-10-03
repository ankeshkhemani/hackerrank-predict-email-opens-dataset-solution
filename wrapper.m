#! /usr/local/bin/octave -qf

%%One-vs-all
%% Initialization
clear all; close all; clc
%% ============ Loading Relevant input Data ============
fprintf('\nLoading relevant_input_data...\n')
inputfile="data/accurate_input_data.mat";
outputfile="data/relevant_input_data.mat";
[X y] = preProcessing(inputfile,1);

m = size(X, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup the parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% alogrithm = 0 for linear Regression, alogrithm= 1 for single class logistic regression, 
% alogrithm >1 for multiclass logistic regression
% learning_curve_points = number of points needed to be plotted.
%learning_curve_stabiliser is number of attempts at each point in learning curve.

global polynomial_degree = 4;
global num_labels = 1;
global fmincgIter = 200;
global test_sample_size = 100;
global learning_curve_stabiliser = 5;
global learning_examples = 40;
global algorithm = 4;
%1=Linear Reg,2=logistic Reg,3=multiclass log Reg,4=NN Classification

global selected_poly =1;
global lambda = 160; %10 or 160
global lambda_vec = [.01 .02 .04 .08 .16 .32 .64 1.32 2.64 5 10 20]';
global input_layer_size  = 58;   % 58 features
global hidden_layer_size = 100;   % 100 hidden units. Change this to check perf.
global num_labels = 1;          % 1 label, 0 or 1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   PLOT LEARNING CURVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% =========== Learning Curve vs examples =============
%[error_train, error_val] = plotLearningExampleCurve(X, y)
%fprintf('Program paused. Press enter to continue.\n');
%clear error_train error_val;
%pause;
%close all;

%% =========== Learning Curve vs polynomial_degree =============

%[error_train, error_val] = plotLearningPolynomialCurve(X, y)
%fprintf('Program paused. Press enter to continue.\n');
%clear error_train error_val;
%pause;
%close all;

%% =========== Validation Curve to find out lamda  =============

%[error_train, error_val] = plotValidationCurve(X, y)
%fprintf('Program paused. Press enter to continue.\n');
%clear error_train error_val;
%pause;pause;pause;
%close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%============== Train and test various models on large datasets===============
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Put in polynomial features
[X_poly] = polyFeatures(X, selected_poly);
[X_norm, mu, sigma] = featureNormalize(X_poly);

%Randomize dataset
n = rand(length(X_norm),1); 
[garbage index] = sort(n); 
X = X_norm(index,:); 
y = y(index,:);

%Take first 60% examples for training
mark1=floor(length(X_poly)* 0.6);
Xtrain = X(1:mark1,:);
ytrain = y(1:mark1,:);

%Take next 20% examples for validation
mark2=floor(length(X_poly)* 0.8);
Xval = X(mark1+1:mark2,:);
yval =y(mark1+1:mark2,:);

%Take next 20% examples for testing
Xtest = X(mark2+1:end,:);
ytest =y(mark2+1:end,:);




###################################
# One-Vs-All
###################################
%Train  Models
fprintf('\nTraining One-vs-All Logistic Regression using Selected Model...\n')
[all_theta] = oneVsAll(X, y, lambda);


%TEST MODEL
fprintf('\nTesting against Validation dataset\n')
probabilities = predictOneVsAll(all_theta, Xval);

%setting pred as output prediction
if (size(probabilities,2) == 1)
	pred=probabilities >=.5;  %in case of only one class


else
	[xdummy, pred] = max (probabilities,[], 2);  %in case of multiple classes
#This line predicts the mail being opened or not.
#probability of mail being opened = mail opened and not unsubscribed + mail opened and unsubscribed
#probability of mail NOT being opened = mail not opened and not unsubscribed + mail not opened and unsubscribed
pred_open = ((probabilities(:,1) + probabilities(:,2)) > (probabilities(:,3) + probabilities(:,4)));

endif
fprintf('\nCross Validation Prediction Accuracy: %f\n', mean(double(pred == yval)) * 100);

probabilities = predictOneVsAll(all_theta, Xtest);
if (size(probabilities,2) == 1)
	pred=probabilities >=.5; 
endif
fprintf('\nTest Prediction Accuracy: %f\n', mean(double(pred == ytest)) * 100);
save data/lcmodel.mat all_theta mu sigma;

###################################
# Neural Networks
###################################


X=Xtrain(1:1000,:);
y=ytrain(1:1000,:);

fprintf('\nTraining Neural Network... \n')

[nn_params] = train(X,y,lambda);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('Program paused. Press enter to continue.\n');
pause;


pred = nnPredict(Theta1, Theta2, Xtest);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100)

fprintf('\nTraining Complete\n')
save data/lcmodel.mat Theta1 Theta2 mu sigma;


