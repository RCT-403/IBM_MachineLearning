import numpy as np 
import pandas as pd
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import MinMaxScaler
import pylab
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import fcluster

pdf = pd.read_csv('cars_clus.csv')

# Change all invalid entries (not integers) into NaN in all the col except the car brand and model
pdf.iloc[:,2:pdf.shape[1]-1] = pdf.iloc[:,2:pdf.shape[1]-1].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)

# Lets select our feature set
X = pdf.iloc[:,7:pdf.shape[1]-2]

# Scale all the values to (0,1)
x = X.values 
feature_mtx = MinMaxScaler().fit_transform(x)

# Create the hierarchy model
D = distance_matrix(x,x) 
Z = hierarchy.linkage(D, 'complete')
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )  
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =6, orientation = 'right')
plt.show()

# Usually, hierarchy has no specified clusters
# But we can add a final restriction where it ends
# e.g. by distance or number of clusters
max_d = 3
k = 5
clusters1 = fcluster(Z, max_d, criterion='distance')
clusters2 = fcluster(Z, k, criterion='maxclust')


