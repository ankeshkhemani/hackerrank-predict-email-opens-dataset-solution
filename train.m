#! /bin/octave -qf
function [theta] = train(X, y, lambda)
%Trains based on the specified variable algorithm and returns theta

% Initialize some useful values
m = length(y); % number of training examples
global algorithm;

if (algorithm == 3)
	theta = oneVsAll(X, y, lambda);
elseif(algorithm==4)
	global input_layer_size;
	global hidden_layer_size;
	global num_labels;
	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
	options = optimset('MaxIter', 200);
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X, y, lambda);
	[theta, cost] = fmincg(costFunction, initial_nn_params, options);
endif

clear m X y lambda;
end