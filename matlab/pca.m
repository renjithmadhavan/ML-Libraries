% Load the data.
data = csvread('../data/ingredients.csv');

% mean feature vector
dataMean = mean(data);

% Subtract off the mean for each dimension.
dataMean = bsxfun(@minus, data, dataMean);
dataCov = dataMean'*dataMean/size(data, 1);

% Do singular value decomposition.
[U,S,V] = svd(dataCov);

% Show transform data into eigenvector basis.
transformedData = (dataMean*U)