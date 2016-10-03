#! /bin/octave -qf
function [error_train, error_val] = plotLearningExampleCurve(X, y)

global lambda;
global learning_examples;
global algorithm;
curve_type = 1;
[error_train, error_val] = learningCurve(X, y, curve_type);

plot(3:learning_examples+2, error_train, 3:learning_examples+2, error_val);
title('Linear Regression Learning curve over examples')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([2 learning_examples+4 0 3])
if (algorithm==3)
	print('plots/lr_vs_examples.png','-dpng','-r300');
elseif (algorithm==4)
	print('plots/NN_vs_examples.png','-dpng','-r300');
endif

clear curve_type X y;
end