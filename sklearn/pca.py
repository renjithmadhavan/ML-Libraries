import numpy as np
import os
from sklearn import decomposition

# Load the data.
f = open(os.path.dirname(__file__) + '../data/ingredients.data')
data = np.fromfile(f, dtype=np.float64, sep=' ')
data = data.reshape(-1, 4)
f.close()

# Perform PCA.
pca = decomposition.PCA(n_components=4)
pca.fit(data)
X = pca.transform(data)

# Show transform data into eigenvector basis.
print X