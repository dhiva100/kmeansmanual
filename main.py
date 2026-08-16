import DatasetHandler
from Kmeans import Kmeans

raw_data, mean, std, min_value, max_value = DatasetHandler.process_dataset(r"C:\Users\ASUS\OneDrive\Documents\Python Projects\pythonProject1\machinelearning\kmeans_manual\kmeansmanual\WineQT.csv")
raw_data = raw_data.drop(columns=["quality", "Id"])
data_property = (mean[0:11], std[0:11], min_value[0:11], max_value[0:11])
data = raw_data.values.tolist()
print("data_property: ", data_property[0])
# kmean = Kmeans()
# # kmean.find_target_k([1, 10], data)
# clusters, centroids = kmean.converge_centroids(data=data, target_k=5, n_iterations=50)
# kmean.visualize_clusters(data, centroids, clusters)