class analysisObj:
    def __init__(self, name, N, transfer_duration, initial_clustering_duration, energy):
        self.name = name
        self.N = N
        self.transfer_duration = transfer_duration
        self.initial_clustering_duration = initial_clustering_duration
        self.energy = energy

    def getName(self):
        return self.name

    def getInitialClusteringDuration(self):
        return self.initial_clustering_duration

    def getTransferDuration(self):
        return self.transfer_duration

    def getEnergy(self):
        return self.energy

    def getN(self):
        return self.N
