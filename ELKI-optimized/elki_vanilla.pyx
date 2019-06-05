#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
cimport cython
from numpy cimport float64_t
from cython cimport view
from sklearn.metrics.pairwise import euclidean_distances
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport fabs
from cpython cimport buffer, array
import array

np.import_array()

ctypedef np.float64_t FLOAT_t
ctypedef np.intp_t INT_t
ctypedef np.ulong_t INDEX_t
ctypedef np.uint8_t BOOL_t

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct Sorter:
    INT_t index
    double value

cdef int _compare(const_void *a, const_void *b):
    cdef double v = ((<Sorter*>a)).value-((<Sorter*>b)).value
    if v < 0: return -1
    if v >= 0: return 1

cdef void cyargsort(double[:] data, Sorter * order) nogil:
    cdef INT_t i
    cdef INT_t n = data.shape[0]
    for i in range(n):
        order[i].index = i
        order[i].value = data[i]
    qsort(<void *> order, n, sizeof(Sorter), _compare)

cdef void argsort(double[:] data, long *order) nogil:
    cdef INT_t i
    cdef INT_t n = data.shape[0]
    cdef Sorter *order_struct = <Sorter *> malloc(n * sizeof(Sorter))
    cyargsort(data, order_struct)
    for i in range(n):
        order[i] = order_struct[i].index
    free(order_struct)
#------------------------------------------------------------------------------------------
cdef struct cdTuple:
    long cluster_id
    double cluster_distance;

cdef struct tupleList:
    cdTuple *tuples
    size_t len

cdef void initTupleList(tupleList * tList,int numClusters) nogil:
    cdef:
        size_t memLen = sizeof(cdTuple) * numClusters
        cdTuple *mem = <cdTuple *> malloc(memLen)
    tList.tuples = mem
    tList.len = numClusters
    memset (mem, 0, memLen)

cdef np.ndarray[double, ndim=2] get_distance_matrix(np.ndarray X, np.ndarray C):
    cdef np.ndarray D
    D = X[:, np.newaxis, :] - C[np.newaxis, :]
    D = np.sum(D**2, axis=2)
    D = np.sqrt(D)
    return D

@cython.cdivision(True)
cdef int calculate_cluster_size(int N, int k):
    return (N + k-1) // k

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tupleList get_best_point_distances(long point_ind, double[:, :] all_distances, double[:] distances_for_point) nogil:
    # do argsort
    cdef:
        long var = 0

    #for var in range(all_distances.shape[1]):
    for var in prange(all_distances.shape[1], nogil=True):
        distances_for_point[var] = all_distances[point_ind, var]

    cdef:
        int tsize = all_distances.shape[1]
        long *order = <long *> malloc(tsize * sizeof(long))
        int j = 0

    #for j in range(tsize):
    for j in prange(tsize, nogil=True):
        order[j] = j

    argsort(distances_for_point, order)
    cdef:
        tupleList cluster_distance_tuple_list
        long i

    initTupleList(&cluster_distance_tuple_list, tsize)
    #for i in range(tsize):
    for i in prange(tsize, nogil=True):
        cluster_distance_tuple_list.tuples[i].cluster_id = order[i]
        cluster_distance_tuple_list.tuples[i].cluster_distance = distances_for_point[order[i]]
    return cluster_distance_tuple_list

cdef cdTuple get_best_cluster_for_point(long point_ind, double[:, :] all_distances, double[:] placeholder) nogil:
    cdef tupleList sorted_point_cluster_distances = get_best_point_distances(point_ind, all_distances, placeholder)
    return sorted_point_cluster_distances.tuples[0]

cdef assign(int[:] labels, double[:] mindist, int N, int k, np.ndarray all_distances_, int max_cluster_size):
    cdef:
        double[:, :] all_distances = all_distances_
        np.ndarray[long, ndim=1] cluster_space_ = np.full(k, max_cluster_size)
        long[:] cluster_space = cluster_space_
        np.ndarray[long, ndim=1] points_inds = np.arange(N)
        tupleList best_point_distances
        long cluster_id
        double dist_to_cluster_center
        long point_ind
        double[:] placeholder = np.empty(all_distances_.shape[1], dtype=np.float64)
        int i
        int j

    for i in range(points_inds.shape[0]):
        point_ind = points_inds[i]
        best_point_distances = get_best_point_distances(point_ind, all_distances, placeholder)
        for j in range(k):
            cluster_id = best_point_distances.tuples[j].cluster_id
            dist_to_cluster_center = best_point_distances.tuples[j].cluster_distance
            if (cluster_space[cluster_id] > 0):
                labels[point_ind] = cluster_id
                mindist[point_ind] = dist_to_cluster_center
                cluster_space[cluster_id] -= 1
                break
    return labels, mindist, cluster_space

cdef double calculateInertia(double[:] arr) nogil:
    cdef:
        double s = 0
        int i = 0
    #for i in range(arr.shape[0]):
    for i in prange(arr.shape[0], nogil=True):
        s += fabs(arr[i])
    return s

cdef long count_size(long *tlist, long max_size) nogil:
    cdef:
        long i = 0
        long count = 0
    #for i in range(max_size):
    for i in prange(max_size, nogil=True):
        if tlist[i] != -1:
            count += 1
    return count

cdef transfer_points(int[:] ilabels, double[:] mindist_, np.ndarray all_distances_, long[:] cluster_space, int N):
    # Prepare transfer list and transfer values
    cdef:
        int[:] labels = np.empty(1, dtype=np.int32)
        double[:] mindist = np.empty(1, dtype=np.float64)
    labels = ilabels
    mindist = mindist_

    cdef:
        long *transfer_list = <long *> malloc(labels.shape[0] * sizeof(long))
        long ti = 0
        double[:, :] all_distances = all_distances_

    for ti in range(labels.shape[0]):
        transfer_list[ti] = -1

    cdef:
        double[:] best_mindist = np.empty((mindist.shape[0]), dtype=np.float64)
        int[:] best_labels = np.empty((labels.shape[0]), dtype=np.int32)
    best_mindist[:] = mindist
    best_labels[:] = labels

    cdef:
        double mindist_sum
        double best_mindist_sum

        long[:] points_by_high_distance = np.argsort(mindist)[::-1]
        int point_cluster
        long swap_cand_ind
        int cand_cluster
        long cluster_id
        double swap_profit
        double dist_to_cluster_center
        double cand_distance_to_cand_cluster
        double cand_distance_to_point_cluster
        double point_distance_to_cand_cluster
        double point_distance_to_point_cluster
        long point_ind
        long i
        long j
        cdTuple cluster_and_distance
        double[:] placeholder = np.empty(all_distances_.shape[1], dtype=np.float64)
        long tl_size = 0

    #for i in prange(points_by_high_distance.shape[0], nogil=True, num_threads=2):
    for i in range(points_by_high_distance.shape[0]):
        point_ind = points_by_high_distance[i]
        point_cluster = labels[point_ind]
        cluster_and_distance = get_best_cluster_for_point(point_ind, all_distances, placeholder)
        cluster_id = cluster_and_distance.cluster_id
        dist_to_cluster_center = cluster_and_distance.cluster_distance

        # assign point to best cluster if possible
        if not cluster_space[cluster_id] <= 0 and point_cluster != cluster_id:
            labels[point_ind] = cluster_id
            mindist[point_ind] = dist_to_cluster_center
            cluster_space[cluster_id] -= 1
            cluster_space[point_cluster] += 1
            best_labels[point_ind] = cluster_id
            best_mindist[point_ind] = dist_to_cluster_center
            continue

        # iterate over transfercandidates and swap if it's appropriate
        for j in range(count_size(transfer_list, labels.shape[0])):
            swap_cand_ind = transfer_list[j]
            cand_cluster = labels[swap_cand_ind]

            if point_cluster != cand_cluster:
                cand_distance_to_cand_cluster = mindist[swap_cand_ind]

                point_distance_to_cand_cluster = all_distances[point_ind, cand_cluster]

                if (point_distance_to_cand_cluster < cand_distance_to_cand_cluster):

                    labels[point_ind] = cand_cluster
                    mindist[point_ind] = all_distances[point_ind, cand_cluster]
                    labels[swap_cand_ind] = point_cluster
                    mindist[swap_cand_ind] = all_distances[swap_cand_ind, point_cluster]

                    mindist_sum = calculateInertia(mindist)
                    best_mindist_sum = calculateInertia(best_mindist)

                    if mindist_sum <  best_mindist_sum:
                        # update the labels since the transfer was a success
                        best_labels[point_ind] = cand_cluster
                        best_mindist[point_ind] = all_distances[point_ind, cand_cluster]
                        best_labels[swap_cand_ind] = point_cluster
                        best_mindist[swap_cand_ind] = all_distances[swap_cand_ind, point_cluster]
                        point_cluster = cand_cluster
                        break

                    else:
                        # reset since the transfer was not a success
                        labels[point_ind] = best_labels[point_ind]
                        mindist[point_ind] = best_mindist[point_ind]
                        labels[swap_cand_ind] = best_labels[swap_cand_ind]
                        mindist[swap_cand_ind] = best_mindist[swap_cand_ind]

        # each point that's not assigned to its best cluster gets added to the transfer_list
        transfer_list[count_size(transfer_list, labels.shape[0])] = point_ind

    free(transfer_list)
    return labels, mindist

def elki(X, k, centroids):
    # measure time
    cdef:
        timespec ts
        timespec te
        double start
        double end
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    """
    vanilla elki algorithm applied to data.
    X - Data matrix
    k - number of clusters
    N - number of Data
    D - number of Features
    centroids - pre initiaslized centroids
    labels - assigned labels for datapoints for pre initiaslized centroids
    all_distances - all distances from points to cluster centers
    """
    cdef:
        long N = X.shape[0]
        int D = X.shape[1]

    # 1. calculate max number of points possible in a cluster
    cdef int max_cluster_size = calculate_cluster_size(N, k)
    cdef np.ndarray all_distances = get_distance_matrix(X, centroids)

    # 2. save for each point the min distance
    cdef:
        labels_ = view.array(shape=(N,), itemsize=sizeof(int), format="i")
        mindist_ = view.array(shape=(N,), itemsize=sizeof(double), format="d")
        int[:] labels = labels_
        double[:] mindist = mindist_

    # 3. assign data points anew based in max cluster size
    cdef:
        long[:] cluster_space
    labels, mindist, cluster_space = assign(labels, mindist, N, k, all_distances, max_cluster_size)

    # 4. do transfer
    labels, mindist = transfer_points(labels, mindist, all_distances, cluster_space, N)

    clock_gettime(CLOCK_REALTIME, &te)
    end = te.tv_sec + (te.tv_nsec / 1000000000.)
    return labels, end-start
