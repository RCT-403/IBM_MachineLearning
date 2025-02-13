import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 

# Create Test set
X, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)

# Train the model (with being able to specify cluster size)
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average').fit(X)
labels = agglom.labels_
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))

# Plot the figure
fig = plt.figure(figsize=(6,4)).add_subplot(111)
for k, col in zip(range(len(set(labels))), colors):

    # Create the mask for the labels
    my_members = (labels == k)
    
    # Plots the datapoints with color col.
    fig.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.', markersize=15)

# Remove the numbers in the side
fig.set_xticks(())
fig.set_yticks(())
plt.show()

# Another way to train the model and show the dendogram of how the points paired 
dist_matrix = distance_matrix(X,X) 
Z = hierarchy.linkage(dist_matrix, 'average')
dendro = hierarchy.dendrogram(Z)
plt.show()