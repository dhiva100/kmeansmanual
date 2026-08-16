import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            #print(f"\tInital Assignment: The {cluster} has {len(self.clusters[cluster])}")
            total+=len(self.clusters[cluster])
        # print("total datapoints assigned: ", total)
            
        for i in range(n_iterations):
            #print("Iteration: ", i)
            old_cluster = self.clusters.copy()
            total_difference = 0
            updated_centroids = self.new_centroid_position(self.training_data)
            self.cluster_datapoints(updated_centroids, self.training_data, target_k)
            total = 0
            for cluster in self.clusters:
                #print(f"\tThe {cluster} has {len(self.clusters[cluster])}")
                total+=len(self.clusters[cluster])
                difference = len(self.clusters[cluster]) - len(old_cluster[cluster])
                total_difference+=abs(difference)
                #print(f"Difference in assignment: ", difference)
            #print("Total difference in assignments: ", total_difference)
            # print("total datapoints assigned: ", total)
        return self.clusters, updated_centroids

    def calculate_wcss(self, cluster_assignment, centroid_pos, data):
        wcss = []
        i = 0
        for centroid, values in cluster_assignment.items():
            distance = 0
            for value in values:
                distance += self.calculate_distance(centroid_pos[i], data[value])
            wcss.append(distance)
            i+=1
        return wcss

    def visualize_clusters(self, data, centroids, cluster_assignment):
        from sklearn.decomposition import PCA

        #cluster values
        i = 0
        clusters = [0]*len(data)
        for centroid, values in cluster_assignment.items():
            for value in values:
                clusters[value] = i
            i+=1
        print("clusters: ", clusters)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data)
        centroids_pca = pca.transform(centroids)
        print("X_pca: ", X_pca)
        print("centroids_pca: ", centroids_pca)
        print("centroids_pca: ", centroids_pca[:, 1])

        plt.scatter(X_pca[:, 0],
                    X_pca[:, 1],
                    c=clusters)

        plt.scatter(centroids_pca[:, 0],
                    centroids_pca[:, 1],
                    marker='X',
                    s=200,
                    c='r')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        plt.show()

    def find_target_k(self, k_range, data):
        k_range = list(k_range)
        k_values = []
        wcss_values = []
        for k in range(k_range[0], k_range[1]):
            print("k value: ", k)
            cluster_assignment, centroid_pos = self.converge_centroids(data, k, 20)
            print("target k clusters: ", cluster_assignment)
            print("centroid positions: ", centroid_pos)
            wcss_values.append(np.sum(self.calculate_wcss(cluster_assignment, centroid_pos, data)))
            k_values.append(k)
        print("wcss_values: ", wcss_values)

        plt.plot(k_values, wcss_values, marker='o')
        plt.ylabel("wcss")
        plt.xlabel("Number of clusters (k)")
        plt.title("Elbow Method")
        plt.xticks(k_values)
        plt.grid(True)
        plt.show()
        self.visualize_clusters(data, centroid_pos, cluster_assignment)



