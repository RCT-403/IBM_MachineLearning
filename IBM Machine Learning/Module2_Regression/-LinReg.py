import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# Read the file
df = pd.read_csv("FuelConsumption.csv")

# Make custom Data frame
cdf = df[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Create train/test set by using randomizer
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train Linear Reg model on test set
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']]) # For Multiple Linear Reg, just add more columns to train_x and test_x
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Terms of regression
coef = regr.coef_[0]                # Array of the coefficients of each of the features
intercept = regr.intercept_[0]      # Outputs the intercept of the regression

# Predict the test set using model
test_x = np.asanyarray(test[['ENGINESIZE']]) 
test_y = np.asanyarray(test[['CO2EMISSIONS']])
predict_y = regr.predict(test_x)

# Calculate the accuracy
mae = np.mean(np.absolute(predict_y - test_y)) 
mse = np.mean((predict_y - test_y) ** 2) 
r2 = r2_score(predict_y, test_y) 

print("Mean Absolute Error: %.2f" %mae)
print("Residual sum of Squares: %.2f" %mse)
print("R2-Score: %.2f" %r2)

# Plot the Regression on the test/train set
plt.scatter(train_x, train_y)
plt.plot(test_x, predict_y, 'r')
plt.show()







