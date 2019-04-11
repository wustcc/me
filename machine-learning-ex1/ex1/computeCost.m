function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m=length(y); % number of training examples
% You need to return the following variables correctly 
J = 0;
predictions=X*theta;
sqrerror=(predictions-y).^2;
J=1/2/m*sum(sqrerror);
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
