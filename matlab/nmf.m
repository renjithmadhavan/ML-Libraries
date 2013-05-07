% Load the data.
Data = csvread('../data/fisheriris_data.csv')

% Compute a nonnegative rank-two approximation
% of the measurements of the fisher iris data.
[W,H] = nnmf(Data, 2, 'algorithm','mult');

% Show the results.
W
H
(W * H)