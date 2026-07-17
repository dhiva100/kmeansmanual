import pandas as pd
import numpy as np

class Kmeans:
    def __init__(self):
        self.training_data = None
        self.centroids = []

    def initialize_centroids(self, data, target_k):
        self.training_data = data
        init_centroids = []
        for i in range(target_k):
            centroid = np.random.randint(0, len(self.training_data))
            init_centroids.append(self.training_data[centroid])
        print("The initial centroid positions are: ", init_centroids)
        return init_centroids
