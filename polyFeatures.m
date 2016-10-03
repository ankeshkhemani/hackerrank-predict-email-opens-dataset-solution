#! /bin/octave -qf
function [X_poly] = polyFeatures(X, p)

if(p==1)
	X_poly=X;
else
	X_poly = zeros(size(X,1), p*size(X,2));
	for i = 1:size(X,2)
    	for j = 1:p
        	X_poly(:,(p * (i-1))+j) = X(:,i) .^j;
    	end
	end
endif




% =========================================================================

end
