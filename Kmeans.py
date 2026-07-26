import pandas as pd
import numpy as np

class Kmeans:
    def __init__(self):
        self.training_data = None
        self.centroids = []
        self.clusters = {}

    def reset_cluster_assignments(self, target_k):
        for i in range(target_k):
            self.clusters[f"cluster_{i}"]= []


    def initialize_centroids(self, data, target_k):
        self.training_data = data
        init_centroids = []
        self.reset_cluster_assignments(target_k)
        for i in range(target_k):
            centroid = np.random.randint(0, len(self.training_data))
            init_centroids.append(self.training_data[centroid])
        print("The initial centroid positions are: ", init_centroids[0])
        return init_centroids
    
    def calculate_distance(self, centroid, datapoint):
        distance = np.sqrt(np.sum((np.array(datapoint) - np.array(centroid))**2))
        return distance
    
    def new_centroid_position(self, data):
        updated_centroids = []
        for centroid, values in self.clusters.items():
            sum_array = np.zeros(len(data[0]))
            n = len(values)
            for value in values:
                sum_array += np.array(data[value])
                # print("data value: ", data[value])
            sum_array = sum_array/n
            updated_centroids.append(sum_array.tolist())
        # print("updated_centroids: ", updated_centroids)
        return updated_centroids
    
    def cluster_datapoints(self, centroids, data, target_k):
        self.reset_cluster_assignments(target_k)
        i = 0
        for data_point in data:
            distances = []
            for centroid in centroids:
                distance = self.calculate_distance(centroid, data_point)
                distances.append(distance)
                # if i==0:
                #     print("centroid values: ", centroid)
            self.clusters[f"cluster_{distances.index(min(distances))}"].append(i)
            i+=1
            
    
    def converge_centroids(self, data, target_k, n_iterations):
        init_centroids = self.initialize_centroids(data, target_k)
        self.cluster_datapoints(init_centroids, self.training_data, target_k)
        total = 0
        for cluster in self.clusters:
            print(f"\tInital Assignment: The {cluster} has {len(self.clusters[cluster])}")
            total+=len(self.clusters[cluster])
        # print("total datapoints assigned: ", total)
            
        for i in range(n_iterations):
            print("Iteration: ", i)
            old_cluster = self.clusters.copy()
            total_difference = 0
            updated_centroids = self.new_centroid_position(self.training_data)
            self.cluster_datapoints(updated_centroids, self.training_data, target_k)
            total = 0
            for cluster in self.clusters:
                print(f"\tThe {cluster} has {len(self.clusters[cluster])}")
                total+=len(self.clusters[cluster])
                difference = len(self.clusters[cluster]) - len(old_cluster[cluster])
                total_difference+=abs(difference)
                print(f"Difference in assignment: ", difference)
            print("Total difference in assignments: ", total_difference)
            # print("total datapoints assigned: ", total)
    
    

