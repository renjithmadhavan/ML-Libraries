% Load the data.
trainData = csvread('../data/fisheriris_data.csv');
trainLabel = csvread('../data/fisheriris_label.csv');

% Naive Bayes classifier.
classifier = NaiveBayes.fit(trainData, trainLabel);

% Predict labels.
labels = classifier.predict(trainData)