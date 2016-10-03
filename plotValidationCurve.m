#! /bin/octave -qf
function [error_train, error_val] = plotValidationCurve(X, y)

global lambda;
global algorithm;
global lambda_vec;

curve_type = 3;

[error_train, error_val] = learningCurve(X, y, curve_type);

plot(lambda_vec, error_train, lambda_vec, error_val);
title('Validation Curve with lambda')
legend('Train', 'Cross Validation')
xlabel('Value of Lambda')
ylabel('Error')
if (algorithm==3)
	print('plots/lambda_validation.png','-dpng','-r300');
endif

clear curve_type X y;
end