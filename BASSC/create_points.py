import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import time

def create_samples(N, D):
    mean = np.ones(D)
    np.random.seed(53)
    cov = np.eye(D)
    cov += 0.5*np.ones((D, D))
    cov -= 0.5*np.eye(D)
    X = np.random.multivariate_normal(mean, cov, size=N)
    # A = np.random.multivariate_normal(mean+10, cov, size=N)
    # return np.concatenate([X, A], axis=0)
    return X

def init_samples(sizes = [20000, 30000, 40000, 50000, 100000, 250000, 500000, 1000000]):
    for i in range(len(sizes)):
        filename = "samples_"+str(sizes[i])+".npz"
        X = create_samples(sizes[i], 2)
        np.savez(filename, data=X)

def plot_data(X, K, mus, colorMap ,L=None):
    plt.figure(figsize=(12,8))
    plt.plot(*mus.T, 'x', color='r', ms=10)
    print(20*"+")
    if L is not None:
        for l in range(K):
        #for l in np.unique(L[L >= 0]):
            idx_o = np.where(L == l+1)[0]
            print(len(idx_o))
            idx_x = np.where(L == -(l+1))[0]
            # print("l: {}, l.shape[0]: {}, -l.shape[0]: {}".format(l, idx_o.shape[0], idx_x.shape[0]))
            clrs = np.array(colorMap.to_rgba(l))[None]
            #print(clrs.shape)
            plt.scatter(*X[idx_o].T, c=clrs, label=l)
            plt.scatter(*X[idx_x].T, marker='x', c=clrs, alpha=0.05)
        """
        idx = np.where(L == 1000)[0]
        E = X[idx]
        np.delete(X,idx)
        plt.scatter(*E.T, c='k')

        for l in np.unique(L):
            idx = np.where(L == l)[0]
            plt.scatter(*X[idx].T, c=colorMap.to_rgba(l))
            """
    else:
        plt.scatter(*X.T)
    plt.legend()
    plt.show()

def createColorMap(max, min = 0):
    rainbow = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=min, vmax = max)
    return cmx.ScalarMappable(norm=cNorm,cmap = rainbow)

if __name__ == "__main__":
    K = 5
    #init_samples(sizes=[10000000])

    colorMap = createColorMap(K)
    F = np.load('samples_10000000.npz')
    X = F['data']
    total_time = 0
    from gmm import EquiKMeans

    model = EquiKMeans(X, K=K)

    #plot_data(X, model.m_0, colorMap)
    #model.train(max_iterations=10)
    #plot_data(X, model.m, model.labels)
    for _ in range(1):
        a = time.time()
        model.train(max_iterations=100)
        total_time += (time.time() - a)
        plot_data(model.X, K, model.m, colorMap, model.labels)
    #plot_data(X, model.m, colorMap, model.labels)
    print(model.reassign_values)
    #print(model.cluster_development.shape)
    #print(model.mu_development.shape)
    print("execution time: {}".format(total_time))

# Clusterzentren fallen nach der ersten Iteration zusammen











