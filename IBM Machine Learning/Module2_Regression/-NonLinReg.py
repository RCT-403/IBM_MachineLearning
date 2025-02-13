import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

df = pd.read_csv("china_gdp.csv")
x_data, y_data = (df["Year"].values, df["Value"].values)

# We find a function that we want to match the graph
# For instance we use a sigmoid function
def mod_sigmoid(x, Beta_1, Beta_2):         
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y

# Scale down our data since plotting and curve_fit cannot work with big numbers
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

# Get the optimized paramters 
param, param_cov = curve_fit(mod_sigmoid, xdata, ydata) 
print(" beta_1 = %f, beta_2 = %f" % (param[0], param[1]))

# Predict based on the model
predict_y = mod_sigmoid(xdata, param[0], param[1])

# Print the accuracy of using Sigmoid
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(ydata - predict_y)))
print("Mean Square Error: %.2f" % np.mean((ydata - predict_y) ** 2))
print("R2-Score: %.2f" % r2_score(ydata, predict_y))

# Print the scatter plot
plt.scatter(x_data, ydata)
plt.plot(x_data, predict_y, 'r')
plt.show()

