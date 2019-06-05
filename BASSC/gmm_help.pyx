#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
cimport cython
#from cython cimport view
#from sklearn.metrics.pairwise import euclidean_distances
#from cython.parallel import prange
#from libc.stdlib cimport malloc, free
#from libc.string cimport memcpy, memset
#from libc.math cimport fabs
#from cpython cimport buffer, array
#import array

def outer_product2(np.ndarray arr1, np.ndarray arr2):
    """
    In our code arr1 and arr2 are usually the same
    """
    cdef:
        np.ndarray res = np.zeros((arr1.shape[0], arr1.shape[1], arr1.shape[1]))
        int i = 0

    for i in range(arr1.shape[0]):
        res[i] = np.outer(arr1[i], arr2[i])
    return res

def outer_product1(np.ndarray arr1, np.ndarray arr2):
    """
    In our code arr1 and arr2 are usually the same
    """
    cdef:
        np.ndarray res = np.zeros((arr1.shape[0], arr1.shape[1], arr1.shape[2], arr1.shape[2]))
        int i = 0
        int j = 0

    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            res[i, j] = np.outer(arr1[i, j], arr2[i, j])
    return res

def mTWm_vec_dot(np.ndarray mu, np.ndarray m_0, np.ndarray W_, int K):
    cdef:
        np.ndarray res = np.zeros(K)
        int k = 0

    _sub = np.subtract(mu, m_0)
    for k in range(K):
        _dot1 = np.dot(W_[k], _sub[k])
        _dot2 = np.dot(_sub[k].T, _dot1)
        res[k] = _dot2

    return res

def xnmTWxnm_vec_dot(np.ndarray X, np.ndarray W_, np.ndarray mu):
    cdef:
        np.ndarray res = np.zeros((X.shape[0], mu.shape[0]))
        int p = 0
        int k = 0
        
    _sub = np.subtract(X[:, np.newaxis], mu[np.newaxis])
    for p in range(res.shape[0]):
        for k in range(res.shape[1]):
            _dot1 = np.dot(W_[k], _sub[p, k])
            _dot2 = np.dot(_sub[p, k].T, _dot1)
            res[p, k] = _dot2
    return res


def tr_w0_W_vec_dot(np.ndarray W_0, np.ndarray W_):
    cdef:
        np.ndarray res = np.zeros(W_.shape[0])
        int i = 0

    for i in range(W_.shape[0]):
        res[i] = np.trace( np.dot(W_0, W_[i]) )

    return res
