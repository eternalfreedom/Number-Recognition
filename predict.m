function p = predict(Theta1, Theta2, X)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% adding 1s in front of Xs for the bias unit
X = [ones(m, 1) X];

% computing hidden layer using Theta1
a = sigmoid(X * Theta1');

% adding bias unit to newly computed hidden layer
a = [ones(m, 1) a];

% output layer using Theta2
h = sigmoid(a * Theta2');

% matrix to store max_prob and output p
% this can also discard max_prob using [~, p], but choosing to keep max_prob for records sake
[max_prob, p] = max(h, [], 2);        % [] is max syntax saying don't take the max of h and anything, 2 refers column vector returning largest in each row

end
