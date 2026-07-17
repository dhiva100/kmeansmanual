import DatasetHandler
from Kmeans import Kmeans

raw_data = DatasetHandler.process_dataset(r"C:\Users\ASUS\OneDrive\Documents\Python Projects\pythonProject1\machinelearning\kmeans_manual\kmeansmanual\WineQT.csv")
data = list(raw_data)
print(data)
print("shape: ", data.shape)
print("len: ", len(data))

kmean = Kmeans()
kmean.initialize_centroids(data, 5)
