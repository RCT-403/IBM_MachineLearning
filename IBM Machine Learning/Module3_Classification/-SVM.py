import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score, jaccard_score

cell_df = pd.read_csv("_CellSamples.csv")

# If we run cell_df.dtypes, we see that one of the columns have some values that are not numerical, so we drop these.

# coerce changes the non-numeric values to NaN and not.null removes these values
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()] 
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int') 

# Create the Feature matrix and vector class, and the other sets
X = np.asarray(cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
y = np.asarray(cell_df['Class'])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# Testing the model, kernel represents the kernelling method
clf = svm.SVC(kernel='rbf').fit(X_train, y_train)
y_predict = clf.predict(X_test)

# Accuracy using f1_score and jaccard_score
f1 = f1_score(y_test, y_predict, average='weighted') 
jac = jaccard_score(y_test, y_predict,pos_label=2)
print("F1-Score: %.2f" %f1)
print("Jaccard Score: %.2f" %jac)
