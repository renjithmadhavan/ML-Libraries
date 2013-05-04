import numpy as np
import os
from shogun.Features import RealFeatures, MulticlassLabels
from shogun.Classifier import GaussianNaiveBayes

# Load the data.
f = open(os.path.dirname(__file__) + '../data/fisheriris.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 4)
f.close()

f = open(os.path.dirname(__file__) + '../data/fisheriris_label.csv')
label = np.fromfile(f, dtype=np.float64, sep=' ')
f.close()

# Naive Bayes classifier.
feat = RealFeatures(data.T)
test = RealFeatures(data.T)
labels = MulticlassLabels(label)
nbc = GaussianNaiveBayes(feat, labels)
nbc.train()

# Predict labels.
results = nbc.apply(test).get_labels()

print results
