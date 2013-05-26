import numpy as np
import os
from sklearn.naive_bayes import MultinomialNB

# Load the data.
f = open(os.path.dirname(__file__) + '../data/fisheriris.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 4)
f.close()

f = open(os.path.dirname(__file__) + '../data/fisheriris_label.csv')
label = np.fromfile(f, dtype=np.float64, sep=' ')
f.close()

# Naive Bayes classifier.
nbc = MultinomialNB()
nbc.fit(data, label)

# Predict labels.
labels = np.matrix(nbc.predict(data)).T
print labels
