import sklearn.datasets
import sklearn.cluster
import scipy.cluster.vq
import matplotlib.pyplot as plot

n = 100
k = 3

# Generate fake data
data, labels = sklearn.datasets.make_blobs(
    n_samples=n, n_features=2, centers=k)

# scipy
means, _ = scipy.cluster.vq.kmeans(data, k, iter=300)

# scikit-learn
kmeans = sklearn.cluster.KMeans(k, max_iter=300)
kmeans.fit(data)
means = kmeans.cluster_centers_

plot.scatter(data[:, 0], data[:, 1], c=labels)
plot.scatter(means[:, 0], means[:, 1], linewidths=2)
plot.show()