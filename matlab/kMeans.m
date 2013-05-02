% Load the data.
Data = csvread('../data/two_cluster.csv');

% Perform kmean with 2 clusters.
[idx,ctrs] = kmeans(Data, 2, 'Distance', 'city', 'Replicates', 5);

% Show cluster association.
idx

% Show cluster centers.
ctrs