import sys
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

df = pd.read_csv('_Drug.csv')

# Output a numpy array w/o the headers
x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df[['Drug']]

# sklearn cannot take in categorical values, so we have to convert it
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M']) # Changes 'F','M' into 0,1,... respectively
x[:,1] = le_sex.transform(x[:,1])  # Changes all of the entries of the second column of the data

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
x[:,2] = le_BP.transform(x[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
x[:,3] = le_Chol.transform(x[:,3]) 

# Get the training data
# Random_State can be anything and is psuedoRandom, so the split is always the same given the same random state value. 
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=3) 

# Uses entropy loss, and depth of 4 nodes
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(train_x, train_y)
predict_y = drugTree.predict(test_x)

# Uses Subset Accuracy (or measures correct/total)
score = metrics.accuracy_score(test_y, predict_y)
print("DecisionTree's Accuracy: %.4f" % score)

# Export the tree as a dot file, then run using "dot -Tpng tree.dot -o tree.png"
export_graphviz(drugTree, out_file='tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
