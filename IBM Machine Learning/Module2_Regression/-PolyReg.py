import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# If we do Polynomial Regression for multiple regression, we need to be careful with Polynomial Features
# i.e, if we use degree 2, and x has 3 features, a,b,c 
# the polynomial features will be [1,a,b,c,a^2,b^2,c^2,ab,bc,ac]
poly = PolynomialFeatures(degree=2)
poly_train_x = poly.fit_transform(train_x)
poly_test_x = poly.fit_transform(test_x)

regr = linear_model.LinearRegression()
regr.fit(poly_train_x, train_y)
predict_y = regr.predict(poly_test_x)

mae = np.mean(np.absolute(predict_y - test_y)) 
mse = np.mean((predict_y - test_y) ** 2) 
r2 = r2_score(predict_y, test_y) 

print("Mean Absolute Error: %.2f" %mae)
print("Residual sum of Squares: %.2f" %mse)
print("R2-Score: %.2f" %r2)

xx = np.arange(0.0, 10.0, 0.1)
yy = regr.intercept_[0]+ regr.coef_[0][1]*xx+ regr.coef_[0][2]*np.power(xx, 2)
plt.scatter(train_x, train_y)
plt.plot(xx,yy, '-r')
plt.show()




