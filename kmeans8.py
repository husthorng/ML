# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=5, random_state=101)
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
plt.scatter(X[:, 0], X[:, 1]);
plt.show();
#print(y);
from sklearn.cluster import KMeans
Cluster = KMeans(n_clusters=5)
Cluster.fit(X)
y_pred = Cluster.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='plasma')
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
plt.show();