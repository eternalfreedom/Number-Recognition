function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);

% don't regularize theta 0 since it is the bias term
theta(1) = 0;  

% cost function                                       ||| + regularized component
J = 1 / m * sum( -y' * log(h) - (1 - y)' * log(1 - h)) + (1 / (2 * m) .* lambda .* (theta' * theta));

% gradient function           ||| + regularized component
grad = [1 / m * (h - y)' * X]' + (lambda .* theta / m);;

grad = grad(:);

end
