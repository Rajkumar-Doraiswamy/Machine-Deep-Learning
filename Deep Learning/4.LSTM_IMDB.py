 """
Last amended: 28th June, 2019
My folder:   /home/ashok/Documents/8.rnn
	        OneDrive/Documents/recurrentNeuralNetwork

Data location: ~/.keras/datasets/imdb.npz
Ref: Page 205, Chapter 6, Deep Learning with Python, by Francois Chollet


Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras
    A sequence is a time series.
	Sequence classification is a predictive modeling problem where you have some
	sequence of inputs over space or time and the task is to predict a category for
	the sequence.
Problem:
	The Large Movie Review Dataset (often referred to as the IMDB dataset)
	contains 25,000 highly-polar movie reviews (good or bad) for training and
	the same amount again for testing. The problem is to determine whether a
	given movie review has a positive or negative sentiment.

Three methods:
    Ist using LSTM only							=>	CC
    IInd using LSTM with Dropout                =>  DD
    IIIrd using Convolution followed by lstm    =>  EE

Files to lookup:
                   i)  keras_embedding_tokenizer.py
                   ii) document_to_id_conversion.py

RAM:  >=4GB

Data Source:
	https://keras.io/datasets/
	Downloaded datasets get stored by default in folder:  ~/.keras/datasets
Ref:
	http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
"""

#########################
#***** AA. Call libraries
#########################
# 1.0
%reset -f
import numpy

# 1.1 Keras provides access to the IMDB dataset built-in.
#     The imdb.load_data() function allows to load the dataset
#     in a format that is ready for use in neural network and
#     deep learning models
from keras.datasets import imdb

# 1.2 Keras models and layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
# 1.3
from keras.layers import LSTM
# 1.4 Transform each word to a vector-space using
#     Embedding layer
from keras.layers.embeddings import Embedding
# 1.5 For padding sequence of texts to equal length coded as integers
from keras.preprocessing import sequence
# 1.6 Miscelleneous
import matplotlib.pyplot as plt
import time

# 1.7 fix random seed for reproducibility
#     Each time ipython is quit and restarted,
#     then whenever X_train[:10] and X_test[:10]
#     are seen they appear to be the same.
#     *************************************************
#     Students may like to check this reproduciability
#     *************************************************
#     This would imply that accuracy of each of the three
#     models below can be compared even if ipython is quit
#     after creating and testing each model (ie CC, DD and EE)
numpy.random.seed(7)


#########################
#***** BB. Process input
#########################


# 2.0 Load data. Will be downloaded if not available locally under ~/.keras/datasets
#     While loading into RAM keep the top-n words that have the maximum freq, zero the rest
#     Data is at: https://s3.amazonaws.com/text-datasets/imdb.npz

top_highFreq_words_to_keep = 10000				# Vocabulary size
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_highFreq_words_to_keep)


# 2.1 Observe data
X_train.shape           # (25000,)
# 2.1.2
X_train[:2]

# 2.2
len(X_train[0])         # 218
# 2.2.1
y_train
numpy.sum(y_train)         # How many are 1's (half of them, 12500)



# 2.3 Truncate or pad input sequences so that max length = max_review_length
#     This is also our sequence length or timesteps

max_review_length = 500        # Max length of any comment

X_train = sequence.pad_sequences(X_train,
                                 maxlen=max_review_length,
								 padding = "pre"
								 )

X_test = sequence.pad_sequences(X_test,
                                maxlen=max_review_length,
								padding = "pre")

# 2.4 Sequence length now?
len(X_train[0])


###########################
#***** CC. Perform Modeling
###########################

# 3.0 Create simple LSTM model

# 3.1 What would be the word-2-vector size?
#     We select a 32-dimension space
embedding_vector_length = 32

# 3.2
model = Sequential()

# 3.3 Embedding Layer: Output shape of the layer will be 500 X 32.
#                      That is everyone of the 500 words is
#                      represented row-by-row as a vector of 32 column-values
# Ref: https://keras.io/layers/embeddings/


model.add(Embedding(
	                input_dim =top_highFreq_words_to_keep,     # Vocabulary size
	                output_dim=embedding_vector_length,        # Decides number of neurons in hidden layer
                                                               # or size of embedding vector
	                input_length= max_review_length            # (optional) Length of each sequence
                                                               # If not specified now, then
                                                               # it is noted at run-time
                                                               # by considering x_train.shape[1]
	                )
         )

model.summary()      # Model will take as input an integer matrix
                     # of size (batch, input_length).
					 # Now model output:  (None, 500, 32)
                     # None: Batch_dimension
					 # 500: Max words in review
                     # 32 : Output vector size



# 3.4 LSTM model with number of cells equal to 'embedding_vector_length'
#     And number of neurons in hidden layer equal to no_of_neurons_in_hidden_state

no_of_neurons_in_hidden_state = 50      # Same as 'ht' in Colah's blog
                                        # Or state in our example '1.simple_rnn.py'

# 3.5 How to calculate LSTM weights. See:
#  https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model
model.add(LSTM(
               units = no_of_neurons_in_hidden_state,
               return_sequences = False
              )
          )

# 3.5.1
model.summary()    # As numer of timesteps are: max_review_length,
                   #   So there is looping around as many times
                   #     Or effectively there are as many LSTM cells
                   #      but weights/parameters of each lstm do not change
                   #       These remain same. For example, change value of
                   #         max_review_length and see that number of
                   #          LSTM parameters do not change


# 3.5.2
# Ref: https://stats.stackexchange.com/questions/288404/how-does-keras-generate-an-lstm-layer-whats-the-dimensionality
model.layers[1].name
model.layers[1].input_shape            # (None, 500, 32)
# Input shape should be: (batch_size, timesteps, input_dim)
# input_dim: dimensionality of the input (integer) or number of features
#  in the input. This value is 32.
#  timesteps  will define how many times your network will "unfold" through time.
#   In our case it is 500.
#    So unfold first time. Get error; adjust weights
#      unfold IInd time, get error; adjust weights


# Output of LSTM layer is from 'no_of_neurons_in_hidden_state'
# All this output goes into the following Dense layer
# Dense output layer with a single neuron and
#  a sigmoid activation function to make 0 or 1 predictions
#   for the two classes (good and bad) in the problem
# 3.6

model.add(Dense(1, activation='sigmoid'))

model.summary()       # Number of parameters increase by 50 + 1

# 3.7 Configure model

model.compile(
	          loss='binary_crossentropy',
	          optimizer='adam',
	          metrics=['accuracy']
	          )


"""
3.8
Embedding layer weights are function of:
      i)  top_highFreq_words_to_keep
      ii) embedding_vecor_length
          = 10000 * 32 = 320000

lstm_2 weights are a function of:
	i)   no_of_neurons_in_hidden_state: 50
	ii)  embedding_vecor_length : 32
	iii) max_review_length : 500

See this: http://deeplearning.net/tutorial/lstm.html

"""


# 4.0 Fit the model

start = time.time()
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,        # Takes 30 minutes. Reduce it to 3 in class
                    batch_size=64,    # Epoch counter jumps by this much at each increment
                    verbose =1
                   )
end = time.time()
print("Time taken: ", (end - start)/60, "minutes")   # Time taken:  9.261374553044638 minutes


# 4.1 Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))        # Accuracy: 85% to 86.97%

# 5.0 Plot how network learns as per epochs
def plot_learning_curve():
    val_acc = history.history['val_acc']
    tr_acc=history.history['acc']
    epochs = range(1, len(val_acc) +1)
    plt.plot(epochs,val_acc, 'b', label = "Validation accu")
    plt.plot(epochs, tr_acc, 'r', label = "Training accu")
    plt.title("Learning Curve: Training and validation accuracy")
    plt.legend()
    plt.show()

# 5.1
plot_learning_curve()



########################################################
#***** DD. LSTM For Sequence Classification With Dropout
########################################################
## Exit ipython. Start again. Process AA and BB
##  JUMP CC and come to DD.


# 5.0 Revised model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(
	                input_dim =top_highFreq_words_to_keep,
	                output_dim=embedding_vecor_length,
	                input_length=max_review_length
	                )
         )

# 5.1 This layer is a new one
model.add(Dropout(0.2))

# 5.2
no_of_neurons_in_hidden_state = 50
model.add(LSTM(units = no_of_neurons_in_hidden_state))

model.add(
           Dropout(0.2)    # Drop 20% of embedding-vector values
         )

model.add(Dense(1, activation='sigmoid'))

model.compile(
	          loss='binary_crossentropy',
	          optimizer='adam',
	          metrics=['accuracy']
	          )

# 5.3
model.summary()

# 6.0  10 minutes on my machine
start = time.time()
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=3,             # Takes 10 minutes
          batch_size=64,
          verbose =1
          )
end = time.time()
print("Time taken: ", (end - start)/60, "minutes")


# 6.1 Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

##################################################################################
#***** EE. LSTM For Sequence Classification With Convolution layer and MaxPooling
##################################################################################
## Exit ipython. Start again. Process AA and BB
##  JUMP both CC and DD and come to EE.
#  Refer: http://deeplearning.net/tutorial/lstm.html
# Create the model

# 7
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(
	                input_dim =top_highFreq_words_to_keep,
	                output_dim=embedding_vector_length,
	                input_length=max_review_length
	                )
         )
# 7.1 This layer is now added. We use 1-D layer
model.add(Conv1D(
	            filters=32,
	            kernel_size=3,
	            padding='same',
	            activation='relu'))

# 7.2 Pooling layer
model.add(MaxPooling1D(pool_size=2))

# 7.3
no_of_neurons_in_hidden_state = 50
model.add(LSTM(units = no_of_neurons_in_hidden_state))
model.add(Dense(1, activation='sigmoid'))

# 7.4
model.compile(
	          loss='binary_crossentropy',
	          optimizer='adam',
	          metrics=['accuracy']
	          )

# 7.5
model.summary()

# 7.6   Five minutes on my machine
start = time.time()
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=3,
          batch_size=64,
          verbose =1
          )
end = time.time()
print("Time taken: ", (end - start)/60, "minutes")

# 8. Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
####################################### DONE #############################################

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
