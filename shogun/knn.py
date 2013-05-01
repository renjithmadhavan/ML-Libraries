import numpy as np
import os
from shogun.Features import RealFeatures, BinaryLabels
from shogun.Classifier import KNN
from shogun.Distance import EuclideanDistance

# Load train data.
f = open(os.path.dirname(__file__) + '../data/arcene_train.data')
trainData = np.fromfile(f, dtype=np.float64, sep=' ')
trainData = trainData.reshape(-1, 10000)
f.close()

f = open(os.path.dirname(__file__) + '../data/arcene_train.label')
trainLabel = np.fromfile(f, dtype=np.int32, sep=' ')
f.close()

# Load test data.
f = open(os.path.dirname(__file__) + '../data/arcene_test.data')
testData = np.fromfile(f, dtype=np.float64, sep=' ')
testData = testData.reshape(-1, 10000)
f.close()

f = open(os.path.dirname(__file__) + '../data/arcene_test.label')
testLabel = np.fromfile(f, dtype=np.float64, sep=' ')
f.close()

# Construct a KNN classifier with a neighborhood size of 9.
feat = RealFeatures(trainData.T)
distance = EuclideanDistance(feat, feat)
labels = BinaryLabels(trainLabel.astype(np.float64))
testFeat = RealFeatures(testData.T)
knn = KNN(9, distance, labels)
knn.train()

# Predict the classification.
output = knn.apply(testFeat).get_labels()

# Validate the classification.
print output == testLabel  
