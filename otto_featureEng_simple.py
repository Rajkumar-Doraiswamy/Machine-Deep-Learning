# -*- coding: utf-8 -*-
"""
Last amended: 8th March, 2019
Myfolder: C:\\Users\\ashok\\OneDrive\\Documents\\xgboost\\otto
          /home/ashok/Documents/8.otto

Ref:
https://www.kaggle.com/alexpengxiao/preprocessing-model-averaging-by-xgb-lgb-1-39
About Random Projection:
    https://turi.com/learn/userguide/feature-engineering/random_projection.html

Objectives:
        i)   Using pandas and sklearn for modeling
        ii)  Feature engineering
                  a) Using statistical measures
                  b) Using Random Projections
                  c) Using clustering
                  d) USing interaction variables
       iii)  Feature selection
                  a) Using derived feature importance from modeling
                  b) Using sklearn FeatureSelection Classes
        iv)  One hot encoding of categorical variables
         v)  Classifciation using Decision Tree and RandomForest

"""

# 1.0 Clear memory
%reset -f

# 1.1 Call data manipulation libraries
import pandas as pd
import numpy as np

# 1.2 Feature creation libraries
from sklearn.random_projection import SparseRandomProjection as sr  # Projection features
from sklearn.cluster import KMeans                    # Cluster features
from sklearn.preprocessing import PolynomialFeatures  # Interaction features

# 1.3 For feature selection
# Ref: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif  # Selection criteria

# 1.4 Data processing
# 1.4.1 Scaling data in various manner
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
# 1.4.2 Transform categorical (integer) to dummy
from sklearn.preprocessing import OneHotEncoder

# 1.5 Splitting data
from sklearn.model_selection import train_test_split

# 1.6 Decision tree modeling
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
# http://scikit-learn.org/stable/modules/tree.html#tree
from sklearn.tree import  DecisionTreeClassifier as dt

# 1.7 RandomForest modeling
from sklearn.ensemble import RandomForestClassifier as rf

# 1.8 Plotting libraries to plot feature importance
import matplotlib.pyplot as plt
import seaborn as sns

# 1.9 Misc
import os, time, gc


################## AA. Reading data from files and exploring ####################

# 2.0 Set working directory and read file
#os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\xgboost\\otto")
os.chdir("D:\\Raj\\Training\\Big Data\\Python\\Ensemble Modeling")
os.listdir()

# 2.1 Read train/test files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2.2 Look at data
train.head(2)
train.shape                        # 61878 X 95
test.shape                         # 114368 X 94

# 2.3 Data types
train.dtypes.value_counts()   # All afeatures re integers except target


# 2.4 Target classes are almost balanced
train.target.value_counts()


# 2.5. Drop column(s) not needed
train.drop(columns = ['id'] , inplace = True)
test.drop(columns = ['id'] ,  inplace = True)
train.shape                # 61878 X 94 Index 0 to 92 are features
                           #         Index 93 is class/target
test.shape                 # 144368 X 93

# 2.6 There is one row in test
#     where all values except id
#     are 0s. We need to drop
#     this row:

# 2.6.1 One of the rows in test dataset is all zeros
#       We need to remove this row
#       Sum each row, and check in which case sum is 0
#       axis = 1 ==> Across columns
x = np.sum(test, axis = 1)
x
v = x.index[x == 0]             # Get index of the row which meets a condition
v

# 2.6.2 Drop this row from test data
test.drop(v, axis = 0, inplace = True)
test.shape                # 114367 X 93


# 3 Check if there are Missing values? None
train.isnull().sum().sum()  # 0
test.isnull().sum().sum()   # 0


############################ BB. Feature Engineering #########################

## i)   Shooting in dark. These features may help or may not help
## ii)  There is no theory as to which features will help
## iii) Fastknn is another method not discussed here

############################################################################
############################ Using Statistical Numbers #####################


#  4. Feature 1: Row sums of features 1:93. More successful
#                when data is binary.

train['sum'] = train.sum(numeric_only = True, axis=1)  # numeric_only= None is default
test['sum'] = test.sum(numeric_only = True,axis=1)

# 4.1 Assume that value of '0' in a cell implies missing feature
#     Transform train and test dataframes
#     replacing '0' with NaN
#     Use pd.replace()
tmp_train = train.replace(0, np.nan)
tmp_test = test.replace(0,np.nan)

# 4.2 Check if tmp_train is same as train or is a view
#     of train? That is check if tmp_train is a deep-copy

tmp_train is train                # False
#tmp_train is train.values.base    # False
tmp_train._is_view                # False


# 4.3 Check if 0 has been replaced by NaN
tmp_train.head(1)
tmp_test.head(1)


# 5. Feature 2 : For every row, how many features exist
#                that is are non-zero/not NaN.
#                Use pd.notna()
tmp_train.notna().head(1)
train["count_not0"] = tmp_train.notna().sum(axis = 1)
test['count_not0'] = tmp_test.notna().sum(axis = 1)
test.head()
train.head()

# 6. Similary create other statistical features
#    Feature 3
#    Pandas has a number of statistical functions
#    Ref: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#computations-descriptive-stats

feat = [ "var", "median", "mean", "std", "max", "min"]
for i in feat:
    train[i] = tmp_train.aggregate(i,  axis =1)
    test[i]  = tmp_test.aggregate(i,axis = 1)


# 7 Delete not needed variables and release memory
del(tmp_train)
del(tmp_test)
gc.collect()


# 7.1 So what do we have finally
train.shape                # 61878 X (1+ 93 + 8) ; 93rd Index is target
train.head(1)
test.shape                 # 144367 X (93 + 8)
test.head(2)


# 8. Before we proceed further, keep target feature separately
target = train['target']
target.tail(2)

# 9.1 And then drop 'target' column from train
#      'test' dataset does not have 'target' col
train.drop(columns = ['target'], inplace = True)
train.shape                # 61878 X 101


# 9.2. Store column names of our data somewhere
#     We will need these later (at the end of this code)
colNames = train.columns.values
colNames



############################################################################
################ Feature creation Using Random Projections ##################
# 10. Random projection is a fast dimensionality reduction feature
#     Also used to look at the structure of data

# 11. Generate features using random projections
#     First stack train and test data, one upon another
tmp = pd.concat([train,test],
                axis = 0,            # Stack one upon another (rbind)
                ignore_index = True
                )


# 12.1
tmp.shape     # 206245 X 101


# 12.2 Transform tmp t0 numpy array
#      Henceforth we will work with array only
tmp = tmp.values
tmp.shape       # (206245, 101)


# 13. Let us create 10 random projections/columns
#     This decision, at present, is arbitrary
NUM_OF_COM = 10

# 13.1 Create an instance of class
rp_instance = sr(n_components = NUM_OF_COM)

# 13.2 fit and transform the (original) dataset
#      Random Projections with desired number
#      of components are returned
rp = rp_instance.fit_transform(tmp[:, :93])

# 13.3 Look at some features
rp[: 5, :  3]


# 13.4 Create some column names for these columns
#      We will use them at the end of this code
rp_col_names = ["r" + str(i) for i in range(10)]
rp_col_names



###############################################################################
############################ Feature creation using kmeans ####################
######################Can be skipped without loss of continuity################


# 14. Before clustering, scale data
# 15.1 Create a StandardScaler instance
se = StandardScaler()
# 15.2 fit() and transform() in one step
tmp = se.fit_transform(tmp)
# 15.3
tmp.shape               # 206245 X 101 (an ndarray)


# 16. Perform kmeans using 93 features.
#     No of centroids is no of classes in the 'target'
centers = target.nunique()    # 9 unique classes
centers               # 9

# 17.1 Begin clustering
start = time.time()

# 17.2 First create object to perform clustering
kmeans = KMeans(n_clusters=centers, # How many
                n_jobs = 2)         # Parallel jobs for n_init



# 17.3 Next train the model on the original data only
kmeans.fit(tmp[:, : 93])

end = time.time()
(end-start)/60.0      # 5 minutes


# 18 Get clusterlabel for each row (data-point)
kmeans.labels_
kmeans.labels_.size   # 206245


# 19. Cluster labels are categorical. So convert them to dummy

# 19.1 Create an instance of OneHotEncoder class
ohe = OneHotEncoder(sparse = False)

# 19.2 Use ohe to learn data
#      ohe.fit(kmeans.labels_)
ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()
                                          # '-1' is a placeholder for actual
# 19.3 Transform data now
dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))
dummy_clusterlabels
dummy_clusterlabels.shape    # 206245 X 9 (as many as there are classes)


# 19.4 We will use the following as names of new nine columns
#      We need them at the end of this code

k_means_names = ["k" + str(i) for i in range(9)]
k_means_names
############################ Interaction features #######################
# 21. Will require lots of memory if we take large number of features
#     Best strategy is to consider only impt features

degree = 2
poly = PolynomialFeatures(degree,                 # Degree 2
                          interaction_only=True,  # Avoid e.g. square(a)
                          include_bias = False    # No constant term
                          )


# 21.1 Consider only first 5 features
#      fit and transform
df =  poly.fit_transform(tmp[:, : 5])


df.shape     # 206245 X 15


# 21.2 Generate some names for these 15 columns
poly_names = [ "poly" + str(i)  for i in range(15)]
poly_names


################# concatenate all features now ##############################

# 22 Append now all generated features together
# 22 Append random projections, kmeans and polynomial features to tmp array

tmp.shape          # 206245 X 101

#  22.1 If variable, 'dummy_clusterlabels', exists, stack kmeans generated
#       columns also else not. 'vars()'' is an inbuilt function in python.
#       All python variables are contained in vars().

if ('dummy_clusterlabels' in vars()):               #
    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])
else:
    tmp = np.hstack([tmp,rp, df])       # No kmeans      <==


tmp.shape          # 206245 X 135   If no kmeans: (206245, 126)


# 22.1 Separate train and test
X = tmp[: train.shape[0], : ]
X.shape                             # 61878 X 135 if no kmeans: (61878, 126)

# 22.2
test = tmp[train.shape[0] :, : ]
test.shape                         # 144367 X 135; if no kmeans: (144367, 126)

# 22.3 Delete tmp
del tmp
gc.collect()


################## Model building #####################


# 23. Split train into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    target,
                                                    test_size = 0.3)

# 23.1
X_train.shape    # 43314 X 135  if no kmeans: (43314, 126)
X_test.shape     # 18564 X 135; if no kmeans: (18564, 126)


# 24 Decision tree classification
# 24.1 Create an instance of class
clf = dt(min_samples_split = 5,
         min_samples_leaf= 5
        )



start = time.time()
# 24.2 Fit/train the object on training data
#      Build model
clf = clf.fit(X_train, y_train)
end = time.time()
(end-start)/60                     # 1 minute

# 24.3 Use model to make predictions
classes = clf.predict(X_test)

# 24.4 Check accuracy
(classes == y_test).sum()/y_test.size      # 72%


# 25. Instantiate RandomForest classifier
clf = rf(n_estimators=50)

# 25.1 Fit/train the object on training data
#      Build model

start = time.time()
clf = clf.fit(X_train, y_train)
end = time.time()
(end-start)/60

# 25.2 Use model to make predictions
classes = clf.predict(X_test)
# 25.3 Check accuracy
(classes == y_test).sum()/y_test.size      # 72%



################## Feature selection #####################

##****************************************
## Using feature importance given by model
##****************************************

# 26. Get feature importance
clf.feature_importances_        # Column-wise feature importance
clf.feature_importances_.size   # 135


# 26.1 To our list of column names, append all other col names
#      generated by random projection, kmeans (onehotencoding)
#      and polynomial features
#      But first check if kmeans was used to generate features

if ('dummy_clusterlabels' in vars()):       # If dummy_clusterlabels labels are defined
    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names
else:
    colNames = colNames = list(colNames) + rp_col_names +  poly_names      # No kmeans      <==

# 26.1.1 So how many columns?
len(colNames)           # 135 with kmeans else 126


# 26.2 Create a dataframe of feature importance and corresponding
#      column names. Sort dataframe by importance of feature
feat_imp = pd.DataFrame({
                   "importance": clf.feature_importances_ ,
                   "featureNames" : colNames
                  }
                 ).sort_values(by = "importance", ascending=False)


feat_imp.shape                   # 135 X 2 ; without kmeans: (126,2)
feat_imp.head(20)


# 26.3 Plot feature importance for first 20 features
g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])
g.set_xticklabels(g.get_xticklabels(),rotation=90)



################ I am done ############################
######### All the code below works  ###################


# 25.4 Select top 93 columns and get their indexes
#      Note that in the selected list few kmeans
#      columns also exist
newindex = feat_imp.index.values[:93]
newindex


# 26 Use these top 93 columns for classification
# 26.1  Create classifier object
clf = dt(min_samples_split = 5, min_samples_leaf= 5)
# 26.2 Traion the object on data
start = time.time()
clf = clf.fit(X_train[: , newindex], y_train)
end = time.time()
(end-start)/60                     # 1 minute


# 26.3  Make prediction
classes = clf.predict(X_test[: , newindex])
# 26.4 Accuracy?
(classes == y_test).sum()/y_test.size      # 72%


#######################################
############ Excellent material below
#######################################

##*****************************************
## Using sklearn feature selection classes
##*****************************************

# 27 Create selection object. No of desired features
NoOfDesiredFeatures = 90
sk = SelectKBest(mutual_info_classif,
                 k= NoOfDesiredFeatures)
# 27.1  Let object learn the data
#       Takes time
sk.fit(X_train, y_train)

# 27.2 Get now the best 'NoOfDesiredFeatures' features
#      For both train and test datasets
X_train_new= sk.transform(X_train)
X_test_new = sk.transform(X_test)

# 27.3
X_train.shape
X_train_new.shape      # 90 best features


# 28 Use these for modeling
# 28.1
clf = dt(min_samples_split = 5, min_samples_leaf= 5)

# 28.2
start = time.time()
clf = clf.fit(X_train_new, y_train)
end = time.time()
(end-start)/60                     # 1 minute


# 28.3
classes = clf.predict(X_test_new)

# 28.4
(classes == y_test).sum()/y_test.size      # 71.8%


############################################  I am done ######################
