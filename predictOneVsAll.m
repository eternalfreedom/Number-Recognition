function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];    

% calculated the cost function while transposing all_theta
h = sigmoid( X * all_theta');

% matrix to store max_prob and output p
% this can also discard max_prob using [~, p], but choosing to keep max_prob for records sake
[max_prob, p] = max(h, [], 2);        % [] is max syntax saying don't take the max of h and anything, 2 refers column vector returning largest in each row

end
