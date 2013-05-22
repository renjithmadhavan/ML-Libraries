% Load the Data.
DataSet = csvread('../data/fisheriris_data.csv');
% Load the labels.
labels = csvread('../data/fisheriris_label.csv');
% Merge the dataset and the labels.
Data = [labels DataSet];

% Using the neighborhood components analysis 
% from the Matlab Toolbox for Dimensionality Reduction
% by Laurens van der Maaten.
% http://homepage.tudelft.nl/19j49/Matlab_Toolbox_for_Dimensionality_Reduction.html 
[X, M] = compute_mapping(Data, 'NCA', 4);

% Show the results.
M.M