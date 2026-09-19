import pandas as pd
import DatasetHandler
from Kmeans import Kmeans

raw_data, mean, std, min_value, max_value = DatasetHandler.process_dataset(r"C:\Users\ASUS\OneDrive\Documents\Python Projects\pythonProject1\machinelearning\kmeans_manual\kmeansmanual\WineQT.csv")
raw_data = raw_data.drop(columns=["quality", "Id"])
data_property = (mean[0:11], std[0:11], min_value[0:11], max_value[0:11])
data = raw_data.values.tolist()
print("data_property: ", data_property[0])
kmean = Kmeans()
# # kmean.find_target_k([1, 10], data)
clusters, centroids = kmean.converge_centroids(data=data, target_k=5, n_iterations=50)
kmean.visualize_clusters(data, centroids, clusters)
original_centroids = DatasetHandler.denormalize_centroid(data_property, centroids)
whole_data = []
i = 0
for centroid_names, values in clusters.items():
    new_centroid = original_centroids[i]
    new_centroid.append(f"cluster_{i}")
    whole_data.append(original_centroids[i])
    for value in values:
        new_row = data[value]
        new_row.append(f"cluster_{i}")
        print("new_row: ", new_row)
        whole_data.append(new_row)
    i+=1
print("whole_data: ", whole_data)
updated_df = pd.DataFrame(whole_data, columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",	"chlorides", "free sulfur dioxide",	"total sulfur dioxide",	"density",	 "pH",	"sulphates", "alcohol", "assigned cluster"])
updated_df.to_csv("clustered_data.csv", index=False)