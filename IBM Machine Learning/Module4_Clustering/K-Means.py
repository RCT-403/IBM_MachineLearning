import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
import pandas as pd
from sklearn.preprocessing import StandardScaler

cust_df = pd.read_csv("Cust_Segmentation.csv")

# Note that Address is categorical so we cannot use this
df = cust_df.drop('Address', axis=1)

# We have Customer ID as the the first entry, but we don't want to touch this
# We also change all NaN values to 0
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

# Training and adding the cluster label into the data frame
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12).fit(X)
labels = k_means.labels_
df["Clus_km"] = labels

# This will help us analyze how the cluster was formed more easily
# e.g. using df.groupby('Clus_km').mean()


# If you wanna graph a 2D graph with the area of the dot representing the level of education
area = np.pi * ( X[:, 1])**2  # Education
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float32), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

# Graph of a 3D figure 
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111,projection='3d')

ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float32))
ax.view_init(elev=15, azim=45)

plt.show()



