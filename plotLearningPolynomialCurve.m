#! /bin/octave -qf
function [error_train, error_val] = plotLearningPolynomialCurve(X, y)

global lambda;
global polynomial_degree;
global algorithm;
curve_type = 2;
[error_train, error_val] = learningCurve(X, y, curve_type);

plot(1:polynomial_degree, error_train, 1:polynomial_degree, error_val);
title('Linear Regression Learning curve over polynomial_degree')
legend('Train', 'Cross Validation')
xlabel('Degree of Polynomial')
ylabel('Error')
axis([0 polynomial_degree+1 0 3])
if (algorithm==3)
	print('plots/lr_vs_polynomials.png','-dpng','-r300');
endif

clear X y curve_type;
end