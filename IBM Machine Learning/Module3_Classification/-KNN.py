import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv("_TeleCust1000t.csv")

# Create Feature Matrix and output vectors
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df['custcat'].values

# Standardize the data in X with 0 mean and unit variance
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print(np.bincount(y_test))

# Train Model and Predict on a given K 
k = 4 
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train) # Creates the classifier and fits it with X_train, y_train
predict_y = neigh.predict(X_test)

# Get the Subset Accuracy (correct/total)
test_acc = metrics.accuracy_score(y_test, predict_y)

# If we want to find the best K, we run over several iterations of K, following the same steps above
test_K = 50   
mean_acc = np.zeros((test_K))
std_acc = np.zeros((test_K))

for n in range(1,test_K + 1):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    predict_y=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, predict_y)
    
    # Bonus: Get SD of the accuracy within each test set (but its just True/False)
    std_acc[n-1]=np.std(predict_y==y_test)/np.sqrt(predict_y.shape[0])

best_K = np.argmax(mean_acc) + 1
print(f"Best accuracy is at K = {best_K}")
print("Accuracy: %f" % mean_acc[best_K - 1])

# Example of how to print the accuracy of each of the K's and its std
plt.plot(range(1,test_K+1),mean_acc,'g')
plt.fill_between(range(1,test_K+1),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,test_K+1),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()