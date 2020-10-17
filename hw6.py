import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.cluster import KMeans

china=load_sample_image("china.jpg")
image=china/255
X=image.reshape(-1,3)
kmeans=KMeans(n_clusters=6).fit(X)
segmented_img=kmeans.cluster_centers_[kmeans.labels_]
segemented_img=segmented_img.reshape(image.shape)
kmeans.cluster_centers_
kiner={}
for k in range(3,10):
	kmeans=KMeans(n_clusters=k)
	kmeans.fit(X)
	kiner[k]=kmeans.inertia_
plt.plot(list(kiner.keys()),list(kiner.values()))
plt.show()
