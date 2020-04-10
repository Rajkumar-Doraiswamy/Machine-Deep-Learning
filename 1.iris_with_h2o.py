"""
Last amended: 3rd March, 2019
My folder: /home/ashok/Documents/6.education_analytics

Objectives:
        i) Experiments in Deeplearning
       ii) Learning to work in h2o


DO NOT EXECUTE THIS CODE IN SPYDER--IT MAY FAIL

Ref:
Machine Learning with python and H2O
   https://www.h2o.ai/wp-content/uploads/2018/01/Python-BOOKLET.pdf
H2o deeplearning (latest) booklet
   http://docs.h2o.ai/h2o/latest-stable/h2o-docs/booklets/DeepLearningBooklet.pdf

"""

# 1.0 Call libraries
%reset -f
import pandas as pd
import h2o
import os
# 1.1
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# 2. Start h2o
h2o.init()

# 3. Change working folder and read iris data
os.chdir("D:\\Raj\\Training\\Big Data\\Python\\Neural Networks")
iris =h2o.import_file("iris_wheader.csv")

# 3.1 Explore
type(iris)           #  h2o.frame.H2OFrame

# 3.2
iris.shape
iris.head()
iris.tail()

# 3.3 Transform target to factor column
iris['C5'] = iris['C5'].asfactor()

# 3.4 How many factor levels this columns has
iris['C5'].levels()

# 4.0 Split the dataset into train/test

train,test = iris.split_frame(ratios= [0.7])
train.shape
test.shape

# 4.1 Instantiate model
dl = H2ODeepLearningEstimator(
                             distribution="multinomial",
                             activation = "RectifierWithDropout",
                             hidden = [32,32,32],
                             input_dropout_ratio=0.2,  # It is just as in Random Forest
						       #  we consider some features for each tree
                                                       #   and in this case for each record 
                             epochs = 100
                             )


# 4.2 Train Deep Learning model and predict on test set
dl.train(
        x=['C1','C2','C3', 'C4'],           # Predictor columns
        y="C5",                             # Target
        training_frame=train                # training data
        )

# 4.3 Make prediction on test data
result = dl.predict(test)
result.head()

# 5.0 Transform 'result' to pandas dataframe
re = result.as_data_frame()
type(re)
re['predict']

# 5.1 Add a new column to this dataframe of actual class values
re['actual'] = test['C5'].as_data_frame().values

# 5.2 Get accuracy
sum( re['predict'] == re['actual'])/re.shape[0]


# 5.3 Column importance:
var_df = pd.DataFrame(dl.varimp(),
             columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])

var_df.head()


################################################################
