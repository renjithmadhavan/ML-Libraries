import numpy as np
import os
from mlpy import Knn

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
clf = Knn(9)
clf.compute(trainData, trainLabel)

# Predict the classification.
output = clf.predict(testData)

# Validate the classification.
print output == testLabel