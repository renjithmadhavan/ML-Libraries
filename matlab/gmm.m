% Load the data.
Data = csvread('../data/mvnrnd_data.csv');

% Calculate mixture of Gaussians.
m = gmdistribution.fit(Data, 2);

% The means.
m.mu
% Covariances; one for each Gaussian.
ComponentCovariances = m.Sigma;
ComponentCovariances(:,:,1)
ComponentCovariances(:,:,2)
% The a priori weights of each Gaussian.
MixtureProportions = m.PComponents;
MixtureProportions