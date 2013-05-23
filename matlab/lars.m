% Load the data.
X = csvread('../data/artificial_data_X.csv');
Y = csvread('../data/artificial_data_Y.csv');

% Fit a cross-validated sequence of models with lasso.
[beta fitinfo] = lasso(X,Y,'CV', 10, 'Lambda', 0);

% Find the Lambda value of the minimal cross-validated mean squared error 
% plus one standard deviation. Examine the MSE and coefficients of the
% fit at that Lambda.
lambda = fitinfo.Index1SE;
fitinfo.MSE(lambda);

% Show the results.
beta(:,lambda)