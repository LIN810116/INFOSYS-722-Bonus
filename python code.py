# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:13:32 2018

@author: Chinchien
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
# dataset
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



