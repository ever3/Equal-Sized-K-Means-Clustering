import numpy as np
import sample_points as samples
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

def initialize(X, k, N, D):
    # initialize randome centroids: later k-means++
    centroids_ind = np.random.choice(N, k)
    centroids = X[centroids_ind]

    # calculate distances
    all_distances = euclidean_distances(X, centroids)

    # matrix with all indices and assigned clusters
    labels = all_distances.argmin(axis=1)

    return centroids, labels, all_distances

def calculate_cluster_size(N, k):
    return (N + k-1) // k

def check_label_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

def get_best_point_distances(point_ind, all_distances):
    distance_cluster_sorted = np.argsort(all_distances[point_ind])
    distances_sorted = all_distances[point_ind][distance_cluster_sorted]
    cluster_distance_tuple_list = list(zip(distance_cluster_sorted, distances_sorted))
    return cluster_distance_tuple_list

def get_best_cluster_for_point(point_ind, all_distances):
    sorted_point_cluster_distances = get_best_point_distances(point_ind, all_distances)
    return sorted_point_cluster_distances[0][0], sorted_point_cluster_distances[0][1]

def assign(labels, mindist, N, k, all_distances, max_cluster_size):
    cluster_space = np.full(k, max_cluster_size) # to count if there is enough space in a cluster to assign a point
    points_ind = np.arange(N)
    for point_ind in points_ind:
        for sorted_distances in get_best_point_distances(point_ind, all_distances):
            cluster_id, dist_to_cluster_center = sorted_distances
            if cluster_space[cluster_id] > 0:
                labels[point_ind] = cluster_id
                mindist[point_ind] = dist_to_cluster_center
                cluster_space[cluster_id] -= 1
                break
    return labels, mindist, cluster_space

def elki(X, k, N, D, centroids):
    """
    vanilla elki algorithm applied to data.
    X - Data matrix
    k - number of clusters
    N - number of Data
    D - number of Features
    centroids - pre initiaslized centroids
    labels - assigned labels for datapoints for pre initiaslized centroids
    all_initial_distances - all distances from points to cluster centers
    """

    print("-")

    # 1. calculate max number of points possible in a cluster
    max_cluster_size = calculate_cluster_size(N, k)
    all_initial_distances = euclidean_distances(X, centroids, squared=True)

    # 2. save for each point the min distances
    # new
    labels = np.empty(N, dtype=np.int32)
    labels.fill(-1)
    mindist = np.empty(N)
    mindist.fill(np.infty)
    # mindist = np.empty(N)
    # mindist = np.choose(labels, all_initial_distances.T)

    # 3. assign data points anew based in max cluster size
    labels, mindist, cluster_space = assign(labels, mindist, N, k, all_initial_distances, max_cluster_size)

    # 4. prepare transferlist and trasfervalues
    transferlist = []
    best_mindist = mindist.copy()
    best_labels = labels.copy()
    points_by_high_distance = np.argsort(mindist)[::-1]

    # 5. do transfer -> see if changing clusters of points makes the overall inertia better
    for point_ind in points_by_high_distance:
        # 5.1 get assigned and best possible cluster for point
        point_cluster = labels[point_ind]
        cluster_id, dist_to_cluster_center = get_best_cluster_for_point(point_ind, all_initial_distances)

        # 5.2 assign point to better cluster if possible
        if not cluster_space[cluster_id] <= 0 and point_cluster != cluster_id:
            labels[point_ind] = cluster_id
            mindist[point_ind] = dist_to_cluster_center
            cluster_space[cluster_id] -= 1
            cluster_space[point_cluster] += 1
            best_labels = labels.copy()
            best_mindist = mindist.copy()
            continue

        # 5.3 iterate over transfercandidates and swap if it's appropriate
        for swap_cand_ind in transferlist:
            cand_cluster = labels[swap_cand_ind]
            if point_cluster != cand_cluster:
                # test
                cand_distance = mindist[swap_cand_ind]

                point_distance = all_initial_distances[point_ind, cand_cluster]
                # print("f1: {}, f2: {}, point_distance<cand_distance: {}".format(f1, f2, point_distance < cand_distance))
                if point_distance < cand_distance:

                    labels[point_ind] = cand_cluster
                    mindist[point_ind] = all_initial_distances[point_ind, cand_cluster]

                    labels[swap_cand_ind] = point_cluster
                    mindist[swap_cand_ind] = all_initial_distances[swap_cand_ind, point_cluster]

                    if np.absolute(mindist).sum() <  np.absolute(best_mindist).sum():
                        #print("f1: {}, f2: {}, swap".format(f1, f2))
                        # update the labels since the transfer was a success
                        best_labels = labels.copy()
                        best_mindist = mindist.copy()
                        point_cluster = cand_cluster
                        #break

                    else:
                        #print("f1: {}, f2: {}, noswap".format(f1, f2))
                        # reset since the transfer was not a success
                        labels = best_labels.copy()
                        mindist = best_mindist.copy()

                """
                # 5.3.1 get distances to clusters
                cand_distance_to_cand_cluster = mindist[swap_cand_ind]
                cand_distance_to_point_cluster = all_initial_distances[swap_cand_ind, point_cluster]
                point_distance_to_cand_cluster = all_initial_distances[point_ind, cand_cluster]
                point_distance_to_point_cluster = all_initial_distances[point_ind, point_cluster]

                # 5.3.2 calculate actual profit of swapping clusters
                swap_profit = (cand_distance_to_cand_cluster + point_distance_to_point_cluster) - (cand_distance_to_point_cluster + point_distance_to_cand_cluster)

                # 5.3.3 swap clusters if profit is positiv
                if swap_profit > 0:
                    labels[point_ind] = cand_cluster
                    mindist[point_ind] = point_distance_to_cand_cluster
                    labels[swap_cand_ind] = point_cluster
                    mindist[swap_cand_ind] = cand_distance_to_point_cluster

                    best_labels = labels.copy()
                    best_mindist = mindist.copy()
                    break"""

        # 5.4 each point that's not assigned to its best cluster get added to the transferlist
        transferlist.append(point_ind)

    return best_labels

if __name__ == "__main__":
        """ 1. create samples """
        N = 500
        D = 2
        X = samples.create_samples(N, D)
        X -= X.mean(axis=0)
        # samples.draw_samples(X)

        """ 2. do initialization process """
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        # display kmeans result
        # samples.draw_samples_with_labels_and_centroids(X, k, labels, centroids)

        """ 3. do elki vanilla """
        best_labels, inertia_after, inertia_before = elki(X, k, N, D, centroids, labels)

        # display transfer result
        samples.draw_samples_with_labels_and_centroids(X, k, best_labels, centroids)

        """ 4. evaluate results """
        print("labels before: {} \n, labels after: {} \n, inertia before transfer: {}, inertia after transfer: {}".format(labels, best_labels, inertia_before, inertia_after))
        print("improvement by {}.".format(inertia_before - inertia_after))
