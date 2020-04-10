# Last amended: 15th March, 2019
# My folder: /home/ashok/Documents/10.higgsBoson
# Ref:
#  https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# Objective:
#	   Learning basics of PCA

# 1.0 Reset memory and call libraries
%reset -f
import pandas as pd
import numpy as np
# 1.1 For scaling data
from sklearn.preprocessing import StandardScaler as ss
# 1.2 Import PCA class
from sklearn.decomposition import PCA
# 1.3 Misc
import os

# 2.0 Read data
os.chdir("/home/ashok/datasets/iris")
# 2.1
df = pd.read_csv("iris_wheader.csv",
                  header = None,                # No header column
                  names = ["c1", "c2", "c3", "c4", "target"]   # Specify column names

# 2.2                  )
df.head(2)
df.shape         # (150, 5)

# 3.0 Separating out the features
X = df.loc[:, "c1" : "c4"].values
X.shape          # (150, 4)

# 3.1 Separating out the target
y = df.loc[:,['target']].values

# 4. Before PCA, data must be standardized
scale = ss()
X = scale.fit_transform(X)

# 5.0 Perform pca
# 5.1 Create PCA object
pca = PCA()
out = pca.fit_transform(X)
out.shape   # 150 X 4

# 6. how much variance has been explained by each column
pca.explained_variance_ratio_    # array([0.72770452, 0.23030523, 0.03683832, 0.00515193])

# 6.1 Get cumulative sum
# 95% of variance explained by first two columns
pca.explained_variance_ratio_.cumsum()  # array([0.72770452, 0.95800975, 0.99484807, 1.        ])

# 6.2 Find correlation between columns
#     round the result to two-decimal places
result = np.corrcoef(out, rowvar = False)
np.around(result, 2)          # Thus correlation between columns is 0
                              # All columns are independent

# 6.3 This was the original dataset
x_result = np.corrcoef(X, rowvar = False)
np.around(x_result, 2)          # Thus correlation between columns is 0


# 7.0 So our reduced dataset is:
final_data = out[:, :2]



###### Can finish here

# 7. Plot the data, after creating a
#    dataframe from arrays
pcdf = pd.DataFrame(
                    data =  final_data,
                    columns = ['pc1', 'pc2']
                    )
# 7.1 Add the 'target' column also
#     after mappig flower-names to integers
pcdf['target'] = df['target'].map({'setosa': 1, "versicolor" : 2, "virginica": 3   })
pcdf.head(2)

# 7.2 Plot a scatter plot
pcdf.plot.scatter(x = "pc1",
                  y = "pc2",
                  c = 'target',
                  colormap='viridis'
                  )
###################### I am done ################
