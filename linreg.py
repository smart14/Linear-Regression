# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:50:21 2018

@author: Lenovo
"""

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import style

import pandas as pd

import sklearn

import warnings

from sklearn import linear_model

from sklearn.model_selection import train_test_split

warnings.simplefilter(action = "ignore",category= FutureWarning)

%matplotlib inline

xs = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
ys = [10,12,20,22,21,25,30,21,32,34,35,30,50,45,55,60,66,64,67,72,74,80,79,84]

len(xs)
len(ys)

plt.scatter(xs,ys)
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.show

# from the above plot, dependent variable is linear with independent variable
# linear regression => y = mx+b
# where m is coefficient and b is intercept
# find m and b
def slope_intercept(x_val,y_val) :
    x=np.array(x_val)
    y=np.array(y_val)
    m= (((np.mean(x)*np.mean(y)) - np.mean(x*y)) /
        ((np.mean(x)*np.mean(x)) - np.mean(x*x)))
    m=round(m,2)
    b = (np.mean(y) - np.mean(x) *m)
    b = round(b,2)
    return m,b    

m,b=slope_intercept(xs,ys)
m
b

reg_line = [(m*x)+b for x in xs]

#plotting regression line
plt.scatter(xs,ys,color="red")
plt.plot(xs,reg_line)
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.title("Making a Regression line")
plt.show()


#RMSE
def rmse(y1,y_hat):
    y_actual = np.array(y1)
    y_pred = np.array(y_hat)
    error = (y_actual - y_pred)**2
    error_mean = round(np.mean(error))
    err_sq = np.sqrt(error_mean)
    return err_sq

rmse(ys,reg_line)


# Linear Regression using Boston Dataset

from sklearn.datasets import load_boston

boston = load_boston()

print(boston.data.shape)

df_x = pd.DataFrame(boston.data,columns=boston.feature_names)

df_y = pd.DataFrame(boston.target)

df_x.head(13)

# To get the list of column names
names=[i for i in list(df_x)]
names

#Calling Linear Regression Model

regr = linear_model.LinearRegression()

# Splitting the data

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)

x_train.head()
y_train.head()
x_test.head()
#fitting the linear regression model to training dataset
regr.fit(x_train,y_train)

#Intercept
regr.intercept_

#Coefficients
regr.coef_

# Mean Squared error
np.mean((regr.predict(x_test)-y_test)**2)

# Explained variance Score: 1 is perfect prediction
regr.score(x_test,y_test)

#Coefficient of Independent variables
regr.coef_[0].tolist()
regr.coef_

#Attaching slopes to respective independent variables
slopes = pd.DataFrame(list(zip(names,regr.coef_[0].tolist())),columns=["names","coefficient"])

slopes

# Plotting Test Values
style.use("bmh")
plt.scatter(regr.predict(x_test),y_test)
plt.scatter(regr.predict(x_test),y_test)

# It is mandtory, choosing only important variables for the model
# Here imp means highly significant
# significant means p value < 0.05 (indicates that variable is of 95% significant)

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Ordinary Least square

model1 = sm.OLS(y_train,x_train)

model1

result = model1.fit()

print(result.summary())

# Removing the variables whose pvalue is greater than 0.05

model2 = sm.OLS(y_train,x_train[['CRIM','ZN','CHAS','RM','DIS','RAD','TAX','PTRATIO','B','LSTAT']])

result2= model2.fit()

print(result2.summary())

# A General approach to compare two different models is AIC and the model with minimum AIC is the best one

# in the above case model2 is the best one

# Dealing with Multicollinearity

# When two or more independent variables in a regression are highly related to one another,then they dont provide unique or independent information to regression

import seaborn # a library similar to matplotlib
corr_df = x_train.corr(method='pearson')
print("------------Create A Correlation Plot--------")
mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True
seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=2.5)

#Show the plot we reorient the labels for each column and row to make them easier to read.
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
