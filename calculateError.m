#! /bin/octave -qf
function error = calculateError(X, y, theta)
% Calculates error in data set based on which algorithm wrapper is using

global algorithm;

global input_layer_size;
global hidden_layer_size;
global num_labels;


error=0;
m=size(X,1);

if (algorithm == 3)
	[error, grad] = lrCostFunction([ones(m, 1) X], y, theta, 0);
elseif(algorithm==4)
	[error, grad] = nnCostFunction(theta, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, 0);
endif

clear X y theta m grad;
end





        

