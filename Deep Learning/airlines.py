"""
Last amended: 6th May, 2018
Myfolder: 		/home/ashok/Documents/12. time_series_data
Datafolder:		/home/ashok/.keras/datasets/airlines_timeseries_data

Ref: 
    https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f


Objective:  
            Predict one-dimensional timeseries data


    Also install pydot and graphviz, as below:

	$ source activate tensorflow


"""



# 1. Call libraries
%reset -f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1.1 Keras models
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM

# 1.2 sklearn libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1.3 Misc (for RMSE etc)
import math, os


################## Data reading ############################

# Data ingestion and processing
# 2.0 
pathToData = "/home/ashok/.keras/datasets/airlines_timeseries_data"
os.chdir(pathToData)

# 2.1 Read data
dataframe = pd.read_csv("international-airline-passengers.csv.zip",
	                    compression='infer', 
	                    usecols=[1],
                        encoding="ISO-8859-1"      # 'utf-8' gives error, hence the choice
                        )

# 2.2 Plot data
plt.plot(dataframe)
plt.show()


# 3.0 Normalize dataset
dataset = dataframe.values
type(dataset)                            # numpy array


# 3.1 Change array values to float for subsequent scaling
dataset = dataset.astype('float32')

# 3.1 Normalize the dataset.
#     We will also use the scaler later for inverse_transform
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
dataset[:10]


# Data splitting
# 4.1 Split into train and test sets in 67% and 33%
train_size = int(len(dataset) * 0.67) 
test_size = len(dataset) - train_size
train_size, test_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
len(train), len(test)



# 5. We define a function to create a new dataset
#    The function takes two arguments: the dataset,
#    which is a NumPy array that we want to convert
#    into two datasets, dataset and the look_back,
#    which is the number of previous time steps to 
#    use as input variables to predict the next time
#    period — in this case defaulted to 1.

#    We create a dataset where trainX is the number
#    of passengers at a given time (t) and trainY is the
#    number of passengers at the next time (t + 1).
#    If lookback is 2, I should get:
#        X= d[0],d[1]    Y = d[2]
#        X= d[92],d[93]  Y = d[94]

# 5.1 Convert an array of values into a dataset matrix
def create_dataset(data, look_back=1):                # len(train) = 96 and let look_back = 2 ; 
	dataX, dataY = [], []                             # Empty arrays
                                                      # Had dataset been of three values, say 1,2,3 
                                                      # then sets would have been [1,2], [2,3]
                                                      # ie [len(data) - lookback -1] = 1, counting from 0
	for i in range(len(data)-look_back-1):            # from 0 to 92 ;  Total 93
		a = data[i:(i+look_back), 0]                  # When i = 0,    a = d[0:2] ie d[0],d[1]
		                                              # When i = 92,   a = d[92:94] ie d[92], d[93]
		dataX.append(a)
		b = data[i + look_back, 0]                    # When i = 0,     b = d[2]
		                                              # When i = 92,    b = d[94]
		dataY.append(b)                               # X= d[0],d[1]    Y = d[2]
	return np.array(dataX), np.array(dataY)           # X= d[92],d[93]  Y = d[94]



"""
def create_dataset(data, look_back=1):                # len(train) = 96 and let look_back = 1 ; 
	dataX, dataY = [], []                             # Empty arrays
	for i in range(len(data)-look_back-1):            # from 0 to 93 ;  Total 94
	                                                     
		a = data[i:(i+look_back), 0]                  # When i =0,    a = d[0:1] ie d[0]
		                                              # When i =93,   a = d[93:94] ie d[93]
		dataX.append(a)
		b = data[i + look_back, 0]                    # When i = 0,    b = d[1] 
		                                              # When i = 93,   b = d[94]
		dataY.append(b)          
	return np.array(dataX), np.array(dataY)

"""


# 5.2 Reshape into X=t and Y=t+1
look_back = 1              #  Try 2
trainX, trainY = create_dataset(train, look_back)
testX, testY   = create_dataset(test, look_back)


# See few points at the beginning
trainX[:15]
trainY[:15]

# See few points at the last datapoints
trainX[ trainX.shape[0] -10 :  ]
trainY[ trainY.shape[0] -10 :  ]


# Examine data shape
trainX.shape
trainY.shape
train.shape


train[93]        # Same as trainX[len(trainX) -1] ; last point of trainX
train[94]        # Same as trainY[len(trainY) -1]



"""
trainX:
array([[0.01544401],
       [0.02702703],
       [0.05405405],
       [0.04826255],
       [0.03281853],
       [0.05984557],
       [0.08494207],

dataset:
array([[0.01544401],
       [0.02702703],
       [0.05405405],
       [0.04826255],
       [0.03281853],
       [0.05984557],
       [0.08494207],

trainY:
array([0.02702703, 0.05405405, 0.04826255, 0.03281853, 0.05984557,
       0.08494207, 0.08494207, 0.06177607, 0.02895753, 0.        ,
       0.02702703, 0.02123553, 0.04247104, 0.07142857, 0.05984557,


"""


# 5.3 Reshape input to be [samples, timesteps, features]
#     What does each mean? See detailed note at the end of this code:
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX.shape             # (94,1,1)    ndarray
testX.shape              # (46,1,1)    ndarray
trainY.shape             # (94,)       ndarray
testY.shape              # (46,)       ndarray



# 6. Create and fit the LSTM network
#    t+1 is being predicted based on t
#    1) LSTM with 4 neurons in the first visible layer
#    2) dropout 20%
#    3) 1 neuron in the output layer for predicting no_of_passengers
#    4) The input shape will be 1 time step with 1 feature (look_back).
#    5) I use the Mean Squared Error (MSE) loss function and the efficient Adam version of stochastic gradient descent.
#    6) The model will be fit for 100 training epochs with a batch size of 70

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))     # Use default activation functions
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# 7. Make predictions
trainPredict = model.predict(trainX)

# 7.1 testX is the data not-seen by model during training
#     But here also time-wise, 46, values are available
testPredict = model.predict(testX)

trainPredict.shape              # (94,1)      94 rows of one value each
testPredict.shape               # (46,1)

trainPredict[:10]
testPredict[:10]


# 8. Invert predictions (rescale back using learned scaler)
#    And actual values
trainPredict = scaler.inverse_transform(trainPredict) 
rescaled_trainY = scaler.inverse_transform([trainY])        # trainY.shape => (94,)

# 8.1 Just examine the data shape/values
trainPredict.shape             # (94,1)       94 rows of one value each
rescaled_trainY.shape          # (1,94)       1 row of 94 values

trainPredict[:10]
rescaled_trainY[0,:10]


# 8.2 Similar actions on test data
testPredict = scaler.inverse_transform(testPredict)
rescaled_testY = scaler.inverse_transform([testY])



# 9. Calculate root mean squared error
trainPredict[:, 0].shape           # Predicted values
rescaled_trainY[0].shape           # Actual values
trainScore = math.sqrt(mean_squared_error(rescaled_trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(rescaled_testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# 10. Shift train predictions for plotting
#     While plotting what is to be kept in mind 
#      is that very first prediction at array-index 0
#       is for the actual value of train data t+lookback periods
#        in future
trainPredictPlot = np.empty_like(dataset)       # Return a new array with the same shape and type as a given array.
trainPredictPlot[:, :] = np.nan                 # dataset.shape: (144,1)
trainPredictPlot.shape                          # (144,1)
trainPredictPlot

# 10.1 As trainPredict is the value of data in future ie at 
#       at t+1 (given t), it has to be plotted/compared with 
#       actual value (at the time).
#        Therefore, to compare prediction with actually what took place
#          we have to shift the predictions to right by look_back.
#            So value of trainPredict at index 0, is actually 
#             to be compared with value of train data at index 1
#              ie index 'look_back' ahead. 
trainPredictPlot[look_back:(len(trainPredict)+look_back), :] = trainPredict
trainPredictPlot[:10]          # Note one nan at the very beginning as a result of 'push-down'
trainPredict[:10]



# 10.2 So how many are not nan in total
np.sum(~np.isnan(trainPredictPlot))               # 94
                                                  # Therefore, including nan at the very first index,
                                                  #   we have a total of 94+1 = 95 slots full.
                                                  #    So testPredict can start at 96th slot.
                                                  #     But this first prediction is nan

# 10.3                                                  
len(trainPredict)+look_back +1                    # 96. But this very first slot should also be nan
                                                  # That is testPredict will get displaced to right
                                                  #  by look_back (prediction for time t is based
                                                  #    on t-look_back values.


# 11. Shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan

# 11.1  testPredict starts where trainPredict ended ie len(trainPredict)+look_back
#       So start is: len(trainPredict)+look_back +1
#       And each prediction shifted to right by look_back
testPredictPlot[((len(trainPredict)+look_back +1) + look_back) :len(dataset)-1, :] = testPredict
testPredictPlot    # First value will be not nan


# 12. Plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

################################################################################
"""
LSTM input shape:
https://github.com/keras-team/keras/issues/2892
https://stackoverflow.com/questions/42532386/how-to-work-with-multiple-inputs-for-lstm-in-keras?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
Example 1
Let us, say, our dataX is like aa and dataY are as below:
    dataX = [['a','b','c'], ['d','e','f'],........['x','f','s'], ['s','e',k']]
    dataY = ['x', 'y', ......'z','r']
    
    dataX, can be re-written as:
        [
             ['a','b','c'],            ==> First input-sequence to model
             ['d','e','f'],            ==> IInd input-sequence to model
             ........
             ['x','f','s']             
             ['s','e','k']             ==> Last input-sequence to model
       ]
        
    Here number of timesteps are three, the length of inner list, ['a','b','c'], 
    all three elements, are used to predict 'x'. Siimilarly 'd','e','f' (all three elements) 
    are used to predict 'y' and so on.
    Per time-unit features are 1. To undetstand features better, see the next example.

Example 2:    
This example has an outer list, an inner list and still an inner list, ie
three nested lists.      
Let us say, our dataX looks as follows:

        [[['b','a','d','x','x','x'],['w','o','r','d','x','x']], [['g','o','o','d','x','x'], ['l', 'e', 't','t', 'e','r']]]
        
        dataY = ['ex', 'dy',...] 

   dataX, can be re-written, as below. Each inner list is one input-sequence to LSTM:
       [
          [
             ['b','a','d','x','x','x'],['w','o','r','d','x','x']
          ],                                              ==> Ist input-sequence to LSTM
          [
             ['g','o','o','d','x','x'], ['l', 'e', 't','t', 'e','r']
          ]                                               ==> IInd input-sequence to LSTM
      ]
       
        
   So, we have two inner lists and per inner list, we have two elements.
   Just, as in the earlier case, numer of elements in the inner list
   determine number of timesteps. So timesteps are: 2
   Per timestep, number of features are: 6
   
Example 3:
Refer file: /home/ashok/Documents/8.rnn/sequenceClassification.py

Model summary is:

Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 500, 32)           153600    
_________________________________________________________________
lstm_1 (LSTM)                (None, 50)                16600     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51        
=================================================================
Total params: 170,251
Trainable params: 170,251
Non-trainable params: 0    


Input shape to LSTM is (None,500,32). This means as follows:
    i)    Total no of samples at this stage are unknown
    ii)   Total number of timesteps are 500. That is 500 lstm cells
    iii)  Total number of features are 32 per input, per timestep
          This indeed is true, for a fake 2-dim word vector would
          be something like [[0.02,0.1], [0.11,0.3]]  for two words.
          
Example 4:
=========
Refer: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
Dataset: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
Consider the problem of Air Pollution prediction, per hour. Data is collected
per-hour basis, as follows:

    No: row number
    year: year of data in this row
    month: month of data in this row
    day: day of data in this row
    hour: hour of data in this row
    pm2.5: PM2.5 concentration
    DEWP: Dew Point
    TEMP: Temperature
    PRES: Pressure
    cbwd: Combined wind direction
    Iws: Cumulated wind speed
    Is: Cumulated hours of snow
    Ir: Cumulated hours of rain

First field is to be dropped. For per-hour prediction, next four fields
are not important, as all data is, in any case, taken on per hour basis.
For LSTM problem, our input wll be:
    pm2.5(t-1) 
    DEWP(t-1) 
    TEMP(t-1) 
    PRES(t-1) 
    cbwd(t-1) 
    Iws(t-1) 
    Is(t-1) 
    Ir(t-1) 
    
Output:
   pm2.5(t)  
    
So input features are eight.   

"""
