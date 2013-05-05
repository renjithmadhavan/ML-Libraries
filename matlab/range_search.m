% Load the data.
X = csvread('../data/rnd_points_large.csv'); 
Y = csvread('../data/rnd_points_small.csv'); 

% Find all neighbors within specified distance using KDTreeSearcher object
NS = KDTreeSearcher(X);
[idx, dist] = rangesearch(NS, Y, 0, 'Distance', 'euclidean')