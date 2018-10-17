# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:13:32 2018

@author: Chinchien
"""

# import libraries
import pandas as pd

# dataset
df = pd.read_csv("marineLife(I4).csv")



df.head(10)
pd.set_option('display.max_columns', 10)

df.shape
df.describe()