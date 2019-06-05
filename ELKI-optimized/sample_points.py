import numpy as np

from sklearn.cluster import KMeans

def create_samples(N, D):
    mean = np.ones(D)
    np.random.seed(53)
    cov = np.eye(D)
    cov += 0.5*np.ones((D, D))
    cov -= 0.5*np.eye(D)
    X = np.random.multivariate_normal(mean, cov, size=N)
    return X

def draw_samples(X):
    import matplotlib.pyplot as plt
    plt.scatter(*X.T)
    plt.show()

def draw_samples_with_labels_and_centroids(X, k, labels, centroids, title):
    import matplotlib.pyplot as plt
    plt.title(title)
    cm = plt.get_cmap("rainbow")
    col = ["b","g","r","c", "m","y","k"]
    for l in np.unique(labels):
        idx = np.where(labels == l)[0]
        X_l = X[idx]
        plt.scatter(*X_l.T, c = col[l])

    """
    c = ["b", "g", "r", "c", "m", "y", "k", "w"]

    for i in range(k):
        cluster_points = np.transpose((labels==i))
        plt.plot(centroids[i][0], centroids[i][1], c[i]+"x")
        plt.plot(X[cluster_points][:,0], X[cluster_points][:,1], c[i]+"*")"""

    plt.show()

if __name__ == '__main__':
    X = create_samples(20, 2)
    draw_samples(X)
