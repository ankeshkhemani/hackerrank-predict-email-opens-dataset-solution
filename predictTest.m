#! /usr/local/bin/octave -qf

%% Initialization
clear all; close all; clc

%% ============ Completing final assignment ============
%fprintf('\nCompleting final assignment...\n')
%fprintf('\nPredicting on given test data...\n')

inputfile="data/accurate_test_data.mat";
outputfile="data/relevant_test_data.mat";

[X] = preProcessing(inputfile,2);

%Apply polyFeatures and normalisation
selected_poly=1;
[X_poly] = polyFeatures(X, selected_poly);
load('data/lcmodel.mat');
X_norm = bsxfun(@minus, X_poly, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);
algorithm=4;

if(algorithm == 3)
	probabilities = predictOneVsAll(all_theta, X_norm);

	%setting pred as output prediction
	if (size(probabilities,2) == 1)
		pred=probabilities >=.5;  %in case of only one class
	endif
elseif(algorithm==4)
	pred=nnPredict(Theta1, Theta2, X);
endif
csvwrite('data/test_output.csv', pred);

fprintf('\nAssignment Complete\n')

#===============================


