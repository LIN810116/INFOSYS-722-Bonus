# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:13:32 2018

@author: Chinchien
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# dataset: please move the csv file into the current working directory
import os
os.getcwd()
df = pd.read_csv("marineLife(I4).csv") 
# settings
# allow to show all columns
pd.options.display.max_columns = None

### Data Understanding

# show dataset
df.head()
# number of row & columns
df.shape
# info
df.info()
df.columns
# data type in pandas
df.dtypes
# general statistics
df.describe()
# distinct values in each column
print("Number of distinct values: ")
print("Class: ", len(df['Class'].unique().tolist()))
print("Order: ", len(df['Order'].unique().tolist()))
print("Family: ", len(df['Family'].unique().tolist()))
print("Genus: ", len(df['Genus'].unique().tolist()))
print("Species: ", len(df['Species'].unique().tolist()))
print("Scientific names: ", len(df['Scientific names'].unique().tolist()))
df.Class.value_counts()
# count missing data
print("Number of missing values: ")
df.isna().sum()
# Outlier check
df.boxplot(column='WMostLong')
df.boxplot(column='EMostLong')
print("number of invalid values in WMostLong: ", len(df[(df.WMostLong > 180) | (df.WMostLong < -180)]))
print("number of invalid values in EMostLong: ", len(df[(df.EMostLong > 180) | (df.EMostLong < -180)]))

### Data Preparation

# new dataset: df_pre: 'Class', 'Order', 'Family', 'WMostLong', 'EMostLong'
df_pre = df[['Class', 'Order', 'Family', 'WMostLong', 'EMostLong']]
# deal with missing values
df_pre = df_pre.dropna(subset=['WMostLong'])
df_pre = df_pre.dropna(subset=['EMostLong'])
df_pre['Class'].fillna ('Unknown', inplace=True)
df_pre['Order'].fillna ('Unknown', inplace=True)
df_pre['Family'].fillna ('Unknown', inplace=True)
print("Number of missing values: ")
df_pre.isna().sum()
df_pre.info()
# create new columns: Coverage, Level & isWide
df_pre['Coverage'] = abs(df_pre.EMostLong - df_pre.WMostLong)
df_pre['Level'] = df_pre.Coverage / 30
df_pre['Level'] = df_pre['Level'].astype(int)
df_pre['isWide'] = df_pre.Coverage > 180
df_pre.dtypes
# create subsets
# subset 1: Class, Order, Family, Coverage
dfCoverage = df_pre[['Class', 'Order', 'Family', 'Coverage']]
# subset 2: Class, Order, Family, Level
dfLevel = df_pre[['Class', 'Order', 'Family', 'Level']]
# subset 3: Class, Order, Family, isWide
dfisWide = df_pre[['Class', 'Order', 'Family', 'isWide']]
# data projection
dfCoverage.describe()
dfLevel.Level.hist()
dfisWide.isWide.hist()

### Modelling

# Generic function
from sklearn import metrics
from sklearn.cross_validation import KFold
def classification_model(model, data, predictors, outcome):
    #fit the model
    model.fit(data[predictors],data[outcome])
    #make predictions on training set
    predictions = model.predict(data[predictors])
    #print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy:%s"%"{0:.3%}".format(accuracy))
    #Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        #filter training data
        train_predictors = (data[predictors].iloc[train,:])
        #the target we're using to train the algorithm
        train_target = data[outcome].iloc[train]
        # Training the algorithm using the predictors and target .
        model.fit(train_predictors, train_target)
        #Record error from each cross−validation run
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    print("Cross−Validation Score: %s" % "{0:.3}".format(np.mean(error)))
    #Fit the model again so that it can be refered outside the function :
    model.fit(data[predictors],data[outcome])

# Encoding datasets
from sklearn import preprocessing
def encode(dataset):
    for Class in dataset.Class:
        if dataset.Class.dtype == type(object):
            le = preprocessing.LabelEncoder()
            dataset.Class = le.fit_transform(dataset.Class)
    for Order in dataset.Order:
        if dataset.Order.dtype == type(object):
            le = preprocessing.LabelEncoder()
            dataset.Order = le.fit_transform(dataset.Order)
    for Family in dataset.Family:
        if dataset.Family.dtype == type(object):
            le = preprocessing.LabelEncoder()
            dataset.Family = le.fit_transform(dataset.Family)
# encode the subsets
encode(dfCoverage)
encode(dfLevel)
encode(dfisWide)
dfCoverage.dtypes

#Coverage

x = dfCoverage[['Class','Order','Family']]
y = dfCoverage['Coverage']
#splitting
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
x_train.shape
x_test.shape
#Ridge regression
from sklearn import linear_model
ridge = linear_model.Ridge(alpha = .5)
ridge.fit(x_train,y_train)
ridge.coef_
y_pred = ridge.predict(x_test)
metrics.explained_variance_score(y_test,y_pred)
metrics.mean_absolute_error(y_test,y_pred)
metrics.mean_squared_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
#SVR
from sklearn import svm
svr = svm.SVR()
svr.fit(x_train,y_train)
y_pred = svr.predict(x_test)
evs = metrics.explained_variance_score(y_test,y_pred)
mae = metrics.mean_absolute_error(y_test,y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print("explained variance score:", evs)
print("mean absolute error:", evs)
print("explained variance score:", evs)
print("explained_variance_score:", evs)
plt.scatter(y_test,y_pred)
plt.xlabel('actual value')
plt.ylabel('predicted value')

# Level

predictor_var = ['Class','Order','Family']
outcome_var = 'Level'
#LinearSVC
from sklearn.svm import LinearSVC
model = LinearSVC(random_state=0, tol=1e-5)
classification_model(model, dfLevel, predictor_var, outcome_var)
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
classification_model(model, dfLevel, predictor_var, outcome_var)

x = dfLevel[['Class','Order','Family']]
y = dfLevel['Level']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
plt.scatter(y_test,y_pred)
plt.xlabel('actual value')
plt.ylabel('predicted value')

# isWide

predictor_var = ['Class','Order','Family']
outcome_var = 'isWide'
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
classification_model(rfc, dfisWide, predictor_var, outcome_var)

x = dfisWide[['Class','Order','Family']]
y = dfisWide['isWide']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
plt.scatter(y_test,y_pred)
plt.xlabel('actual value')
plt.ylabel('predicted value')
print("Feature importance in two-class RandomForestClassifier")
print("Class:", model.feature_importances_[0])
print("Order:", model.feature_importances_[1])
print("Family:", model.feature_importances_[2])
#RandomForestClassifier: iteration2
predictor_var = ['Order','Family']
outcome_var = 'isWide'
rfc = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
classification_model(rfc, dfisWide, predictor_var, outcome_var)
#RandomForestClassifier: iteration3
predictor_var = ['Order']
outcome_var = 'isWide'
rfc = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
classification_model(rfc, dfisWide, predictor_var, outcome_var)

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
classification_model(gbc, dfisWide, predictor_var, outcome_var)

#MLPClassifier
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=10)
classification_model(mlpc, dfisWide, predictor_var, outcome_var)

