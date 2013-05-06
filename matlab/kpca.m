% Load the data.
Data = dlmread('../data/new_circle.txt');

% Show data.
figure(1);
scatter(Data(:,1), Data(:,2)); 
title('Original dataset'); drawnow

% Using the Kernel PCA function by Ambarish Jash.
% http://www.mathworks.com/matlabcentral/fileexchange/27319-kernel-pca
% With sigma = 0.5 instead of 1 and 
% one_mat = ones(size(K))./size(data_in,2) instead of
% one_mat = ones(size(K));
transformedData = kernelpca_tutorial(Data', 2)';

% Show transformed data.
figure(2)
scatter(transformedData(:,1), transformedData(:,2))
title('Result of dimensionality reduction')