import DatasetHandler
from Kmeans import Kmeans

raw_data = DatasetHandler.process_dataset(r"C:\Users\aravi\OneDrive\Documents\Dhiva's Files\kmeans\kmeansmanual\WineQT.csv")
raw_data = raw_data.drop(columns=["quality", "Id"])
data = raw_data.values.tolist()

kmean = Kmeans()
centroids = kmean.converge_centroids(data=data, target_k=5, n_iterations=50)
