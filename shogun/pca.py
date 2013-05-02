import numpy as np
import os
from shogun.Features import RealFeatures, BinaryLabels
from shogun.Classifier import PCA
from shogun.Distance import EuclideanDistance

# Load the data.
f = open(os.path.dirname(__file__) + '../data/ingredients.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 4)
f.close()

# Perform PCA.
feat = RealFeatures(data.T)
preprocessor = PCA()
preprocessor.set_target_dim(4)
preprocessor.init(feat)
preprocessor.apply_to_feature_matrix(feat)

# Show transform data into eigenvector basis.
print np.matrix(feat)




