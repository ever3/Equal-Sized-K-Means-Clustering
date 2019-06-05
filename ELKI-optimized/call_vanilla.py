import numpy as np
import sample_points as samples
import elki_vanilla as elki
import sample_analysis as sa
from sklearn.cluster import KMeans
from time import time
from datetime import datetime
from python_elki_vanilla import elki as elki_py
import os

import kmeans

def writeResultsToFile(duration_dict, D, k):
    systime = datetime.now().strftime("%I%M%S")
    file_name = "result_"
    file_name += str(D) + "_" + str(k) + '_' + systime
    off1 = 9
    off2 = 8
    off_ = 100

    with open("results/"+file_name+".txt", "w+") as f:
        f.write("### Auswertung des elki-Algorithmus ### \n")
        f.write("\n")
        f.write("Randbedingungen: \n")
        f.write("D : " + str(D) + "\n")
        f.write("k : " + str(k) + "\n")
        f.write("\n")
        f.write(off_*"-"+"\n")
        f.write("N" + off1*" " + "initial_duration" + off1*" " + "transfer_duration"+ off1*" "+ "total_duration" + off1*" "+ "energy" + "\n")
        f.write(off_*"-"+"\n")
        for key in duration_dict:
            obj = duration_dict[key]
            totalDuration = obj.getInitialClusteringDuration() + obj.getTransferDuration()
            line = str(obj.getN()) + off2*" " + str(obj.getInitialClusteringDuration()) + off2*" "
            line += str(obj.getTransferDuration()) + off2*" " + str(totalDuration) + off2*" "
            line += str(obj.getEnergy()) + "\n"
            f.write(line)
        f.write(off_*"-"+"\n")

def calculateEnergy(X, centroids, labels):
    energy = 0
    for point in range(X.shape[0]):
        point_dist = X[point] - centroids[labels[point]]
        point_dist = np.sum(point_dist**2)
        point_dist = np.sqrt(point_dist)
        energy += point_dist
    energy /= X.shape[0]
    return energy

if __name__ == '__main__':
    """ 1. initialize properties """
    sizes = [20000, 30000, 40000]                       # ,100000, 1000000, 10000000
    D = 2                                       # number of dimensions
    k = 5                                       # number of clusters
    reps = 1

    duration_dict = {}                          # saves duration for every size

    """ 2. iterate over sizes and do elki """
    for r in range(reps):
        for i in range(len(sizes)):
            """ 2.1 load data from .txt """
            file_name = 'data/samples_'+str(sizes[i])+'.txt'
            X = np.genfromtxt(file_name, dtype='float64')

            """ 2.2 do fast kmeans for cenroids """
            st = time()
            initial_clustering = kmeans.get_clustering(X=X, n_clusters=k, algorithm='auto', verbose=0)
            initial_clustering_duration = time() - st
            centroids = initial_clustering['C']

            """ 2.3 do elki and save duration and energy in dict """
            best_labels, duration = elki.elki(X, k, centroids)
            #st = time()
            #best_labels = elki_py(X, k, sizes[i], D, centroids)
            #duration = time() - st

            energy = calculateEnergy(X, centroids, best_labels)
            anaObj = sa.analysisObj(file_name, X.shape[0], duration, initial_clustering_duration, energy)
            duration_dict[sizes[i]+r] = anaObj # changed sizes[i] -> r

            """ 2.4 draw results (Optional) """
            #if (r == reps-1):
                #samples.draw_samples_with_labels_and_centroids(X, k, best_labels, centroids, file_name)

    """ 3. save time needed and boundary conditions in file """
    writeResultsToFile(duration_dict, D, k)
