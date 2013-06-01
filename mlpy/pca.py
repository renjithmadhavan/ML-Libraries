import numpy as np
import os
import mlpy

# Due to an error (AttributeError: 'module' object has no attribute 'PCA').
# I have manually added a modified class file.
from mlpyPCA import PCA

# Load the data.
f = open(os.path.dirname(__file__) + '../data/ingredients.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 4)
f.close()

# Perform PCA.
pca = PCA()
pca.learn(data)
transformedData = pca.transform(data)

# Show transform data into eigenvector basis.
print transformedData