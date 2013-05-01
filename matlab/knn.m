% Load train data.
trainData = dlmread('../data/arcene_train.data');
trainLabel = dlmread('../data/arcene_train.label');

% Load test data.
testData = dlmread('../data/arcene_test.data');
testlabel = dlmread('../data/arcene_test.label');

% Construct a KNN classifier with a neighborhood size of 9.
knn_classifier = ClassificationKNN.fit(trainData, trainLabel, 'NumNeighbors', 9);

% Predict the classification.
prediction = predict(knn_classifier, testData);

% Validate the classification.
prediction == testlabel