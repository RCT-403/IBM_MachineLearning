import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 

# Make fake set
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12).fit(X)
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = k_means.labels_

# Create a plot and colors set
ax = plt.figure(figsize=(6, 4)).add_subplot(1,1,1) 
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

for k, col in zip(range(len(set(k_means_labels))), colors):

    # Create the mask for the labels
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x,y ticks (the numbers in the side)
ax.set_xticks(())
ax.set_yticks(())

# Show the plot
plt.show()