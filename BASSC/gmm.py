#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import gmm_help as gh
import elki_vanilla_modified as evm
from scipy.special import digamma
from scipy.special import loggamma
from scipy.special import gamma
from scipy.special import logsumexp
#from torch.utils.data import DataLoader
#from dataset import MyDataset

class EquiKMeans():
    """
    Equal-sized k-means mit Inferenzen und variational lower bound
    """
    def __init__(self, X, K, threshold = 1e-3): # k in capital
        self.X = X
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]
        self.K = K
        self.labels = np.full(self.N, 0)
        #self.F = self.K * 10
        self.cluster_size = self.N / self.K
        self.cluster_space = np.full(self.K, self.cluster_size)
        self.unassigned_point_limit = 10  # nur temporär
        self.dln2pi = self.D * np.log(2 * np.pi)
        self._dln2 = self.D*np.log(2)

        self.threshold = threshold

        #self.Cov = np.ones((self.K,self.D,self.D))*np.eye(self.D)[np.newaxis]

        self.beta_0 = 1 #self.N/self.K
        self.Beta_ = self.N * np.ones(self.K)/self.K + self.beta_0
        self.v_0 = self.D + 1
        self.v_ = np.full(self.K, self.v_0)
        self.W_0 = np.identity(self.D)
        self.W_ = np.ones((self.K, self.D, self.D)) * self.W_0[None]
        self.W_0_inv = self.W_0

        #  Expected number of points in each cluster
        self.alpha_0 = np.ones(self.K) * self.N
        self.alpha_ = self.alpha_0[...]
        self.pi = np.ones(self.K)/self.K

        self.resp = np.ones((self.N, self.K))/self.K
        #self.resp = np.asarray([np.random.dirichlet(np.ones(self.K)) for i in range(self.N)])

        self.initialize_centroids()
        self.b0_m0 = self.beta_0 * self.m_0  # is that right? in R^K

        #self.E_ln_Lambda = np.zeros(self.K)      # Exp value of the log-precision
        self.comp_E_ln_Lambda()
        #self.E_ln_pi = np.zeros(self.K)          # Exp value of the mixture proportions
        self.comp_E_ln_pi()
        #self.E_mu_Lambda = np.zeros((self.N, self.K))
        self.comp_E_mu_Lambda()

        self.update_resp()
        #self.estimate_labels()
        self.direction = 0 # bound direction to see if results are getting better or worse
        self.alpha_values = np.zeros(self.K)
        self.soft_assignments = np.zeros(self.K)
        self.hard_assignments = np.zeros(self.K)
        self.points_in_order = np.empty((self.K,), dtype=object)
        self.reassign_values = []


    def get_distances_to_center(self, m):
        return np.min(np.square(np.linalg.norm(self.X - m[:, np.newaxis], axis=2)), axis=0) # DONE numpy optimiert

    def select_next_center(self, minDToC):
        probs = minDToC/minDToC.sum() # berechne für jeden Punkt die Wahrscheinlichkeit als cluster genommen zu werden basierend auf der Entfernung
        cumprobs = probs.cumsum()
        r = np.random.random() # choose r to select a random parameter
        ind = np.where(cumprobs >= r)[0][0]
        return self.X[ind]

    def initialize_centroids(self):
        """
        Function to initialize cluster centers with kmeans++
        """
        np.random.seed(33)
        r = np.random.randint(self.X.shape[0])

        m = self.X[r][np.newaxis]
        while m.shape[0] < self.K:
            minDToC = self.get_distances_to_center(m)
            next_point = self.select_next_center(minDToC)
            m = np.append(m, next_point[np.newaxis], axis=0)

        self.m_0 = m
        self.m = self.m_0[...]
        # print(10*"-"+"kmeans++ finished"+10*"-")

    def calculate_lnC(self, alpha):
        _sum = np.sum(alpha)
        _result = loggamma(_sum) - np.sum(loggamma(alpha))
        return _result

    def update_N_(self):
        """
        Bishop page 477 (10.51)
        """
        self.N_ = self.resp.sum(axis=0)

    def update_X_(self):
        """
        Bishop page 477 (10.52)
        """
        #               N,K,1                       N,1,D
        _prod = self.resp[:, :, np.newaxis] * self.X[:, np.newaxis]
        self.X_ = np.sum(_prod, axis=0) / self.N_[:, None]

    def update_S_(self):
        """
        Bishop page 477 (10.53)
        """
        #                         N,1,D             1,K,D
        _sub = np.subtract(self.X[:, np.newaxis], self.X_[np.newaxis])
        #           N,K,1   N,K,D
        _outer = self.resp[:, :, None, None] * gh.outer_product1(_sub, _sub)
        self.S_ = np.sum(_outer, axis=0) / self.N_[:, None, None]

    def update_alpha_(self):
        """
        Bishop page 478 (10.58)
        """
        self.alpha_ = self.alpha_0 + self.N_

    def compute_statistics(self):
        """
        Function to update N_, X_ and S_, alpha_
        """
        #print("update_statistics")
        # update N_
        self.update_N_()
        # update X_
        self.update_X_()
        # update S
        self.update_S_()
        # update alpha
        self.update_alpha_()

    def update_Beta_(self):
        """
        Bishop page 478 (10.60)
        """
        self.Beta_ = self.N_ + self.beta_0

    def update_m_(self):
        """
        Bishop page 478 (10.61)
        """
        self.m = (self.b0_m0 + self.N_[:, np.newaxis] * self.X_) / self.Beta_[:, np.newaxis]
        pass

    def update_W_(self):
        """
        Bishop page 478 (10.62)
        """

        #           K,D                 K,D
        _sub = np.subtract(self.X_, self.m_0)
        _outer = gh.outer_product2(_sub, _sub) # K,D,D
        _NS_prod = self.N_[:, np.newaxis, np.newaxis] * self.S_ # K,D,D
        _division = (self.beta_0 * self.N_)/(self.beta_0 + self.N_)
        self.W_ = np.linalg.inv(self.W_0_inv[np.newaxis] + (_NS_prod + (_division[:, np.newaxis, np.newaxis] * _outer)))

        #print(self.W_ == W_test_inv)


    def update_v_(self):
        """
        Bishop page 478 (10.63)
        """
        self.v_ = self.v_0 + self.N_ + 1

    def update_parameters(self):
        """
        Function to update Beta_, m_, W_ and v_
        """
        #print("update_parameters")
        # calculate Beta_
        self.update_Beta_()
        # calculate m_
        self.update_m_()
        # calculate W_
        self.update_W_()
        # calculate v_
        self.update_v_()

    def comp_E_ln_Lambda(self):
        """
        Bishop page 478 (10.65)
        """
        sgn, vals = np.linalg.slogdet(self.W_)
        res = sgn*vals
        _dln_logdet_sum = self._dln2 + res
        _digamma_sum = np.sum(digamma((self.v_[:, np.newaxis] - np.arange(1, self.D+1)[np.newaxis] + 1) * 0.5), axis=1)
        self.E_ln_Lambda = _digamma_sum + _dln_logdet_sum
        pass

    def comp_E_ln_pi(self):
        """
        Bishop page 478 (10.66)
        """
        alpha_sum = np.sum(self.alpha_)
        digamma_alpha_sum = digamma(alpha_sum)
        self.E_ln_pi = digamma(self.alpha_) - digamma_alpha_sum
        pass

    def comp_E_mu_Lambda(self):
        """
        Bishop page 478 (10.64)
        """
        _dot = gh.xnmTWxnm_vec_dot(self.X, self.W_, self.m)
        _mult = self.v_ * _dot
        _result = self.D/self.Beta_ + _mult
        self.E_mu_Lambda = _result

    def update_expectations(self):
        """
        Functions to update expectations erw_Prec, erw_pi
        """
        #print("update_expectations")
        # calculate erw_Prec
        self.comp_E_ln_Lambda()
        # calculate erw_pi
        self.comp_E_ln_pi()
        # calculate erw_pi_Prec
        self.comp_E_mu_Lambda()

    def update_resp(self):
        """
        Update responsibilities. Bishop page 477 (10.49)
        """
        #print("update_responsibilities")
        #               K                   K               1
        _al = self.E_ln_pi + 0.5 * (self.E_ln_Lambda - self.dln2pi - self.E_mu_Lambda)
        sum_al = logsumexp(_al, 1)
        self.resp = np.exp(_al - sum_al[:, None])

    def estimate_labels(self, idx):
        """
        Estimate the labels using the responsibilities calculated in advance.
        For now just the highest value for each element is taken.
        """
        #print("estimate_labels")
        _labels = np.argmax(self.resp[idx], axis=1) + 1
        self.labels = _labels

    def assign_points(self, i):
        """
        1. Fix points to clusters if the probability is high, that they
            belong to the cluster
        """

        self.estimate_labels(np.arange(self.N))
        for l in np.unique(self.labels):
            idx_l = np.where(self.labels == l)[0]
            k = np.argmax(self.resp[idx_l], axis=1)[0]
            self.soft_assignments[k] = idx_l.shape[0]  # set soft_assignments

            idx_h = idx_l[self.resp[idx_l, k] > 0.3] # idx only geq threshold

            srt_idx = np.argsort(self.resp[idx_h, k])[::-1]
            idx_h = idx_l[srt_idx[:int(self.cluster_size)]]

            self.alpha_values[k] = idx_h.shape[0]
            self.hard_assignments[k] = idx_h.shape[0]   # set hard_assignments
            seq = self.resp[idx_h, k].argsort()[::-1]   # sequence for assignment_order
            self.points_in_order[k] = idx_h[seq]        # die actual indices in correct order

            # Just for evaluation
            #self.cluster_development[i, k] = idx1.shape[0]  # set cluster_development for evaluation of the algorithm
            #self.mu_development[i, k] = self.m[k]
            #print(" k: {} -> feste size: {}, allgemeine size: {}".format(k+1, idx.shape[0], idx1.shape[0]))

            # setze resp von Werten eines Clusters die über dem Schwellwert sind auf 1. Die resp der anderen Cluster auf 0
            self.resp[idx_h, :] = 0 # alle auf 0
            self.resp[idx_h, k] = 0.9 # ausgewählte auf 1
            self.labels[idx_h] = -l
        #print(40*"-")

        self.alpha_0 = (i+1)*(self.cluster_size - self.hard_assignments)
        for k in range(self.K):
            print("K:{}, H:{}, S:{}".format(k, self.hard_assignments[k], self.soft_assignments[k]))

        #rest_assign = np.sum(self.soft_assignments - self.hard_assignments)
        rest_assign = self.N - np.sum(self.hard_assignments)
        self.reassign_values.append(rest_assign)
        print("R", rest_assign)
        print('--' * 20)
        return

    def call_elki(self):
        cluster_space = np.full((self.K), self.cluster_size)
        points_soft_idx = np.where(self.labels > 0)[0]
        X_of_unassigned = self.X[points_soft_idx]

        for k in range(self.K):
            if self.hard_assignments[k] <= self.cluster_size:
                cluster_space[k] -= self.hard_assignments[k]
            else:
                cluster_space[k] = 0
                to_reassign = self.hard_assignments[k] - self.cluster_size
                reassign_idx = self.points_in_order[k][-int(to_reassign):]
                points_soft_idx = np.concatenate((points_soft_idx, reassign_idx))
                X_of_unassigned = np.concatenate((X_of_unassigned, self.X[reassign_idx]), axis=0)

        """
        print("number of to assign: {}, cluster_space: {}, cluster_space_sum: {}".format(X_of_unassigned.shape[0], cluster_space, cluster_space.sum()))
        print("indices: {}".format(points_soft_idx.shape))
        print("X[indices] == X_of_unassigned: {}".format(X_of_unassigned == self.X[points_soft_idx]))
        """
        # call elki
        unassigned_labels = evm.elki(X_of_unassigned, self.K, self.m, cluster_space.astype(int))
        self.labels[points_soft_idx] = -1 * (np.asarray(unassigned_labels) + 1)
        unique, counts = np.unique(self.labels, return_counts=True)
        print(dict(zip(unique, counts)))
        return

    def isELKIable(self):
        rest_assign = np.sum(self.soft_assignments - self.hard_assignments)
        for k in range(self.K):
            if self.hard_assignments[k] > self.N/self.K:
                rest_assign += self.hard_assignments[k] - self.N/self.K
        self.reassign_values.append(rest_assign)
        #print(rest_assign)
        #unique, counts = np.unique(self.labels, return_counts=True)
        #space_dict = dict(zip(unique, counts))
        #print(space_dict)
        #print(40*"-")
        #if rest_assign <= 30000:
        #    print("elki with {} points".format(rest_assign))
        #    return True
        return False

    def init_evaluate_variables(self, max_iterations):
        # Paramter um den Algorithmus auszuwerten
        self.cluster_development = np.zeros((max_iterations, self.K))       # anzahl zugewieser punkte zu einem cluster
        self.mu_development = np.zeros((max_iterations, self.K, self.D))    # veränderung der Mittelwerte

    def remember_last_values(self):
        self.N_last = np.copy(self.N_)
        self.X_last = np.copy(self.X_)
        self.S_last = np.copy(self.S_)
        self.alpha_last = np.copy(self.alpha_)
        self.Beta_last = np.copy(self.Beta_)
        self.m_last = np.copy(self.m)
        self.v_last = np.copy(self.v_)
        self.W_last = np.copy(self.W_)
        self.E_mu_Lambda_last = np.copy(self.E_mu_Lambda)
        self.E_ln_Lambda_last = np.copy(self.E_ln_Lambda)
        self.E_ln_pi_last = np.copy(self.E_ln_pi)
        self.resp_last = np.copy(self.resp)
        self.soft_assignments_last = np.copy(self.soft_assignments)
        self.hard_assignments_last = np.copy(self.hard_assignments)
        self.reassign_values_last = self.reassign_values.copy()
        self.labels_last = np.copy(self.labels)

    def reset_to_last(self):
        self.N_ = self.N_last
        self.X_ = self.X_last
        self.S_ = self.S_last
        self.alpha_ = self.alpha_last
        self.Beta_ = self.Beta_last
        self.m = self.m_last
        self.v_ = self.v_last
        self.W_ = self.W_last
        self.E_mu_Lambda = self.E_mu_Lambda_last
        self.E_ln_Lambda = self.E_ln_Lambda_last
        self.E_ln_pi = self.E_ln_pi_last
        self.resp = self.resp_last
        self.soft_assignments = self.soft_assignments_last
        self.hard_assignments = self.hard_assignments_last
        self.reassign_values = self.reassign_values_last
        self.labels = self.labels_last

    def equify(self):
        # Weise jeden Punkt seiner Wahrscheinlichkeit nach label hinzu
        self.labels = np.argmax(self.resp, axis=1)
        unique, counts = np.unique(self.labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
        clusters_too_big = np.where(counts > self.N/self.K)[0]
        clusters_too_small = np.where(counts < self.N/self.K)[0]
        """ Step 1 finished """

        # Lösche aus jedem Cluster die überflüssigen Punkte
        for k in clusters_too_big:
            idx = np.where(self.labels == k)[0]
            to_delete_count = int(idx.shape[0] - (self.N/self.K))
            seq = np.argsort(self.resp[idx, k])
            idx_sorted = idx[seq]
            to_delete_idx = idx_sorted[:to_delete_count]
            self.X = np.delete(self.X, to_delete_idx, 0)
            self.labels = np.delete(self.labels, to_delete_idx, 0)
            self.resp = np.delete(self.resp, to_delete_idx, 0)
        """ Step 2 finished """

        # Füge jedem Cluster mit zu wenig Punkten weitere hinzu
        for k in clusters_too_small:
            to_create_count = int(self.N/self.K - count_dict[k])
            Cov_beta = 0.3 * self.W_[k]
            new_points = np.random.multivariate_normal(self.m[k], Cov_beta, size=to_create_count)
            self.X = np.concatenate((self.X, new_points), axis=0)
            single_resp = np.zeros(self.K)
            single_resp[k] = 1
            new_resps = np.zeros((to_create_count, self.K)) + single_resp
            self.resp = np.concatenate((self.resp, new_resps), axis=0)
            new_labels = np.full(to_create_count, k)
            self.labels = np.concatenate((self.labels, new_labels))

        unique, counts = np.unique(self.labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
        self.labels += 1
        return


    def train(self, max_iterations=100):
        self.init_evaluate_variables(max_iterations)
        i = 0
        while(True):
            i += 1
            self.compute_statistics()
            self.update_parameters()
            self.update_expectations()
            self.update_resp()

            self.remember_last_values()
            self.assign_points(i-1)

            if i > 1:
                if (self.reassign_values[-1] - self.reassign_values[-2]) > 0 and i > 1:
                    self.reset_to_last()
                    print("converged")
                    break

            if(i >= max_iterations):
                break
            #if (self.isELKIable() or :# or self.count_unassigned() <= self.unassigned_point_limit ):
            #    break
        self.equify()
        #self.estimate_labels()
        #self.call_elki()


"""
to_assign_idx = indeces_of_unassigned[_idx]             # get true indices of points to be assigned
        to_assign_probs = np.max(assign_probs[_idx], axis=1)    # get best probs of points to be assigned
        to_assign_clusters = np.argmax(assign_probs[_idx], axis=1) # get indices of best probs

        unique, counts = np.unique(to_assign_clusters, return_counts=True) # count how many points want to be assigned in each cluster
        space_dict = dict(zip(unique, counts))

        for k in range(self.K):
            if (self.cluster_space[k] <= 0 or not k in space_dict): # if no point wants to be assigned in this cluster or the cluster is full -> skip
                continue
            isSpace = self.cluster_space[k] - space_dict[k] > -1  # is there enough space to assign all
            if (isSpace):
                # if there is enough space, assign the points
                idx_new = to_assign_idx[to_assign_clusters == k]
                self.labels[idx_new] = k
                # update cluster space
                self.cluster_space[k] -= space_dict[k]
                # update alpha_0
                self.alpha_0[k] -= space_dict[k]
            else:
                # TODO: test this part
                # if there is not enough space
                # -> assign as many points as there is still space
                #    and assign them based on their probabilities
                rest_space = self.cluster_space[k]                       # useless
                idx_new = to_assign_idx[to_assign_clusters == k]
                probs_new = to_assign_probs[to_assign_clusters == k]
                seq = probs_new.argsort()
                sort_idx = idx_new[seq]
                to_assign = sort_idx[:rest_space]
                self.labels[to_assign] = k
                # update cluster space
                self.cluster_space[k] -= rest_space
                # update alpha_0
                self.alpha_0[k] -= rest_space
"""

"""
 
    # values of unassigned points
    idx_of_unassigned = np.where(self.labels == -1)[0]
    X_of_unassigned = self.X[np.where(self.labels == -1)[0]]

    # remaining space in clusters
    cluster_space = np.full(self.K, self.cluster_size)
    unique, counts = np.unique(self.labels, return_counts=True)
    space_dict = dict(zip(unique, counts))
    print(space_dict)

    for key, value in space_dict.iteritems():
        if (key != -1):
            cluster_space[key] -= value

    # call elki
    #print("labels_free: {}, cluster_space: {}".format(self.labels[self.labels == -1], cluster_space))
    print(X_of_unassigned.shape)
    print(cluster_space)
    unassigned_labels = evm.elki(X_of_unassigned, self.K, self.m, cluster_space)
    self.labels[idx_of_unassigned] = unassigned_labels
    unique, counts = np.unique(self.labels, return_counts=True)
    print(dict(zip(unique, counts)))
       
"""
