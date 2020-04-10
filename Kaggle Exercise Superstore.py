# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:53:37 2019

@author: doraisr
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib qt5
os.chdir("D:\\Raj\\Training\\Big Data\\Python\\Kaggle")
os.listdir()
pd.options.display.max_columns = 200
df = pd.read_csv("superstore_dataset2011-2015.csv",encoding = "ISO-8859-1")
df.index
df.columns
df.columns.values
df.describe()
df.dtypes
df.info()
df.count()
df.shape
df.isna()
df['Segment'].value_counts()  
x=df.groupby(['Segment','Customer ID']).count()
x=df.groupby(['Customer ID'])['Profit'].sum()
list(x)
q=df.groupby(['Segment'])['Profit'].aggregate(np.sum).reset_index().sort_values('Profit',ascending=False).head(20)
q.index
q.blocks
bm=q._data
bm.blocks
q.dtypes
x.dtypes
type(q)
type(x)
q
############
w=df.groupby(['Segment','Customer Name'])['Profit'].aggregate(np.sum).reset_index().sort_values(['Segment','Profit'],ascending=False)
w
l=df['Segment'=="Consumer"].count()
l
e=(df.groupby(['Segment'])("Segment'== Consumer)
e
e.head()
pd.DataFrame.

##########
dfl = pd.DataFrame(np.random.randn(5, 4),
   ....:                    columns=list('ABCD'))
dfl       
