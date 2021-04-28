#!/usr/bin/env python
# coding: utf-8

# # Applying Linear,Lasso ,Ridge ,Elastic net Regression on Boston Dataset

# 
# A)Content :
# Data description The Boston data frame has 506 rows and 14 columns.
# 
# This data frame contains the following columns:
# 
# crim per capita crime rate by town.
# 
# zn proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# indus proportion of non-retail business acres per town.
# 
# chas Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 
# nox nitrogen oxides concentration (parts per 10 million).
# 
# rm average number of rooms per dwelling.
# 
# age proportion of owner-occupied units built prior to 1940.
# 
# dis weighted mean of distances to five Boston employment centres.
# 
# rad index of accessibility to radial highways.
# 
# tax full-value property-tax rate per $10,000.
# 
# ptratio pupil-teacher ratio by town.
# 
# black 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# 
# lstat lower status of the population (percent).
# 
# medv median value of owner-occupied homes in $1000s.
# 
# B)Procedure:
# 1.Importing necessary libraries
# 2.Importing boston data set
# 3.Getting the r squared & adjusted r squared value
# 4.Analysis of the data set.
# 5.Analysis of r squared & adjusted r squared value through Lasso regression
# 6.Analysis of r squared & adjusted r squared value through Ridge regression
# 7.Analysis of r squared & adjusted r squared value through Elastic net regression.
# 8.Conclusion

# In[1]:


#importing Required library 
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston


# In[2]:


#importing the data base
df=load_boston()
df


# In[3]:


dataset = pd.DataFrame(df.data)
print(dataset.head())


# In[4]:


dataset.columns=df.feature_names


# In[5]:


dataset.head()


# In[6]:


df.target


# In[7]:


df.target.shape


# In[8]:


dataset["Price"]=df.target


# In[9]:


X=dataset.iloc[:,:-1] ## independent features
y=dataset.iloc[:,-1] ## dependent features


# In[10]:


#importing Required library 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #model 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import pickle #Save the model in HDD

sns.set()


# In[11]:


# Scaling of the dataset
scaler =StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[12]:


X_scaled


# In[13]:


# Splitting the data function
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.30,random_state=355)


# In[14]:


#Interpreting Linear regression 
regression = LinearRegression()

regression.fit(x_train,y_train)


# In[15]:


#Predicting the linear regression
predy=regression.predict(x_test)


# In[16]:


#Interpreting the root mean squared error
rmse=sqrt(mean_squared_error(y_test,predy))


# In[17]:


# root mean Squared error
rmse


# In[18]:


#Interpreting the r squared
r_squared=r2_score(y_test, predy)


# In[19]:


r_squared


# In[20]:


#Interpreting the adjusted r squared
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)


# In[21]:


adjusted_r_squared


# In[22]:


# saving the model to the local file system
filename = 'finalized_model.pickle'
pickle.dump(regression, open(filename, 'wb'))  #final model for deployment


# # Regularization

# In[23]:


#importing Required library 
def adj_r2(x,y,model):
    r2 = model.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# Linear Regression

# In[24]:


# R2 Value for training dataset
regression.score(x_train,y_train)


# In[25]:


#adjusted R2 Value for training dataset
adj_r2(x_train,y_train,regression)


# R2 Value is 72% and adjusted R2 value is 70% for training dataset

# In[26]:


#Test Accuracy (r-Square value)
regression.score(x_test,y_test)


# In[27]:


#Test Accuracy (Adjusted r-Square value)
adj_r2(x_test,y_test,regression)


# R Square value is 76% and adjusted R2 value is 74% for testing dataset

# In[28]:


# Mean Squared error 
mean_squared_error(y_test,regression.predict(x_test))


# # Lasso Regression

# In[29]:


#importing Required library 

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100,200,300,1.1,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.33]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# Best alpha value is 0.33 & MSE is 22.4130

# In[30]:


# Value of lasso_alpha
lasso_alpha=0.33


# In[31]:


# Applying Lasso regrression on training data set
lasso_reg = Lasso(lasso_alpha)
lasso_reg.fit(x_train, y_train)


# In[32]:


# R Squared
r2=lasso_reg.score(x_train,y_train)
print(r2)


# In[33]:


# Adjusted R squared for training data set
adj_r2(x_train,y_train,lasso_reg)


# In[34]:


# Adjusted R squared for testing data set
adj_r2(x_test,y_test,lasso_reg)


# # Ridge Regression

# In[35]:


#importing Required library 

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100,101,220,219,215,210,207,208,206,205]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[36]:


#best alpha value
Ridge_alpha=205 


# In[37]:


# Applying Ridge regrression on training data set
ridge_reg = Lasso(Ridge_alpha)
ridge_reg.fit(x_train, y_train)


# In[38]:


# R Squared
r2=ridge_reg.score(x_train,y_train)
print(r2)


# In[39]:


# Adjusted R squared for training data set
adj_r2(x_train,y_train,ridge_reg)


# In[40]:


# Adjusted R squared for testing data set
adj_r2(x_test,y_test,ridge_reg)


# # Elastic net

# In[41]:


#importing Required library 

from sklearn.linear_model  import  ElasticNetCV
elasticCV = ElasticNetCV(alphas = None, cv =5)

elasticCV.fit(x_train, y_train)


# In[42]:


# Value of Elastic alpha
elastic_alpha=elasticCV.alpha_
elastic_alpha


# In[43]:


#importing Required library 
from sklearn.linear_model  import  ElasticNet
elasticnet_reg = ElasticNet(alpha = elastic_alpha)
elasticnet_reg.fit(x_train, y_train)


# In[44]:


# Elastic Regression of training data set
elasticnet_reg.score(x_train, y_train)


# In[45]:


# Elastic Regression of testing data set
elasticnet_reg.score(x_test, y_test)


# In[46]:


# Adjusted R squared of training data set
adj_r2(x_train,y_train,elasticnet_reg)


# In[47]:


# Adjusted R squared of testing data set
adj_r2(x_test, y_test,elasticnet_reg)


# # Conclusion
# The Elastic Net regularization give better adjusted R2 squared value than other regularization.
# After analysis the model has not been well trained over training data set .
# After analysis for the model there is huge variations in the adjusted r2 squared value for the different regularization methods.
# The accuracy of the model is not good as trained for training data set.
