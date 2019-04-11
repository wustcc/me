function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

predictions=X*theta;
sqrerror=(predictions-y).^2;
J=1/2/m*sum(sqrerror);
%����ȥ��theta0
theta_removeth0=theta(2:end,:);
J=J+lambda/2/m*sum(theta_removeth0.^2);

%�����ݶ�
grad(1,1) = 1 / m * (X(:,1)' * (predictions- y));
grad(2:end,end) = (1 / m * (X(:,2:end)' * (predictions - y)) + lambda / m * theta(2:end));
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
