import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
import itertools 

churn_df = pd.read_csv("_ChurnData.csv")
# pick features for modeling
df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]

X = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X = preprocessing.StandardScaler().fit(X).transform(X)
y = np.asarray(churn_df['churn']).astype('int')

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# Remember that for logistic regression, we input a lin reg model into a sigmoid function
# C gives the inverse inverse of the regularization strength, Solver is to choose the loss function
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
predict_y = LR.predict(X_test)
probab_y = LR.predict_proba(X_test) # Gives a 2d array, first col gives prob of y=0 and second col gives prob of y=1

# Accuracy using Jaccards Index (intersect/union)
from sklearn.metrics import jaccard_score
jac_score = jaccard_score(y_test, predict_y ,pos_label=0)
print("Jaccard_score: %.2f" % jac_score)

# Example if you compute log_Loss
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train, y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print("LogLoss: %.2f" % log_loss(y_test, yhat_prob2))

# Another way to measure the accuracy is by the confusion matrix
def plot_confusion_matrix(cm, classes, # This is just to graph the Confusion Matrix when u input the matrix
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predict_y, labels=[1,0])

# Plot non-normalized confusion matrix and the classification report (f1-score, recall, precision)
print(classification_report(y_test, predict_y))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()


