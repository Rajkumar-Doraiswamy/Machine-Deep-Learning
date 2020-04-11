"""
Last amended: 06/05/2018
Myfolder: /home/ashok/Documents/13. text_generation
Weights folder on Google drive: http://203.122.28.230/moodle/mod/url/view.php?id=2035
Ref: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/#comment-436362

Reshape Input for Long Short-Term Memory Networks in Keras
https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

Objective:
          Text generation using small LSTM Network
		  Text Patterns from Alice in Wonderland

This file is Part I of two files (rather 3).
Files are:          
            i.   text_model_creation.py   => Develop model for text-prediction. Learn patterns
                                             of book 'Alice in Wornderland'
            ii)  text_generation.py       => Use learning to predict text, given a seed text
                                             from the book
            iii) crisp.py                 => Same as (i) but without comments for quick model dvt
                                             in adavance of the scheduled class.
                                             
                                             
Steps:
             i)  Read the book
            ii)  Collect list of unique characters from the book (47)
           iii)  Transform each one of these unique characters into unique integers
                 (character to integer transformation) 
           iv)  Create an empty list, say, dataX      
           iv)  Scan the book from the beginning. Read first 100 letters
                to create a list. Append this list to dataX.
                Right-shift by one character. Get another pattern of 100 letters
                as a list and append it to dataX  (dataX is a list of lists)
                and so on collect patterns till book end.
            v) Each time I read a pattern of 100 characters, also note the
               101st character and store it in dataY. So dataY will have as
               much length as dataX.
            vi) Plan for prediction using these:
                          No of timesteps to use for prediction: 100
                          Features per timestep: 1
                          (see explanation below)
            v)  Build your LSTM model accordingly. You can decide how many
                neurons you want in hidden layer (ht). We have used 256.              


"""


# 1.0 Call libraries
%reset -f
import numpy as np
import sys

# 2. Keras model & layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# 2.1 Store model status information after each epoch
#     espeically model weights along with monitoring
#     parameter
from keras.callbacks import ModelCheckpoint

# 2.2 Needed for to_categorical()
#     to convert target to one-hot-encoded form
from keras.utils import np_utils

# 2.3 Misc
import glob

#############################################################################
##################### AA. Read and process data file ########################
#############################################################################

# 3. Read and process Data file
# Load text file and covert to lowercase
filename = "/home/ashok/.keras/datasets/alice/wonderland.txt"
book_text = open(filename).read()


# 3.1 Convert text to lower case
book_text = book_text.lower()
book_text


# 3.2 Get sorted list of all unique characters in the text
unique_chars = sorted(list(set(book_text)))              # set() retains unique characters
unique_chars 							    			# Try set("ashok sharma")


# 3.3 Have a look at unique_chars
np.array(unique_chars)               # Transform unique_chars to array for easy display
len(np.array(unique_chars))          # Total 47


# 3.4 Create mapping of unique_chars to integers
#     mapping is saved to dictionary of characters vs numbers
char_to_int=dict()            # Empty dictionary
for i,c in enumerate(unique_chars):
	char_to_int[c]= i        # For key c set value of i

# 3.5
char_to_int


"""
Sample of dictionary: char_to_int:
{'\n': 0,
 ' ': 1,
 '!': 2,
 '(': 3,
 ')': 4,

 'm': 29,
 'n': 30,
 'o': 31,
 'p': 32,

Try, one by one to understand:

set("rt, erter wer  wer rwer yutyaS")
list(set("rt erter wer  wer rwer yutyaS"))
sorted(list(set("rt erter wer  wer rwer yutyaS")))
x = sorted(list(set("rt erter wer  wer rwer yutyaS")))
list(enumerate(x))
list(dict((c, i) for i, c in enumerate(x)))
dict(dict((c, i) for i, c in enumerate(x)))


"""


# 3.6 Summarize the loaded data
total_no_chars = len(book_text)          # 144413    len("qweqw rwer")
n_vocab = len(unique_chars)              # 47 => Our target values or predictors       


# 4. Scan the book from beginning till end, 100 chars at a time.
#    First scan first 100 chars, then slide one char to right
#    scan another 100 and so on till the end. Each scan becomes
#    a pattern (list of characters) and will be saved in dataX.
#    The 101st character of every scan will be saved in dataY.
#    dataX is, therefore, a list of lists. dataY is simple list.
#    After each scan, transform each character to corresponding
#    inerger number using char_to_int dictionary.
lstm_seq_len = 100                      # Seq length of characters
                                        # Our LSTM will have as many blocks/cells
dataX = []                              # Input sequence list
dataY = []								# Expected Output list per sequence


# 5. If there are 102 characters, then we will scan
#    2 times ie 102 - 100 = 2. range(2) results in i = 0,1
for i in range(0, total_no_chars - lstm_seq_len, 1):   # Say i = 0
	seq_in = book_text[i:i + lstm_seq_len]             # seq_in  = book_text[0:100] 
	seq_out = book_text[i + lstm_seq_len]              # seq_out = book_text[100] 
	# 5.1 Transform every character in 'seq_in' to integer
	#     And append it to list, dataX
	dataX.append([char_to_int[char] for char in seq_in])  # List comprehension
	# 5.2 Transform target to integer
	dataY.append(char_to_int[seq_out])



# 6. So what is our list length? How many patterns
dataX[0]                               # Note dataX is a list of lists
n_patterns = len(dataX)                # 144313 (100 less than the total)
                                       # For if there are 102 characters
                                       # we can have: 102 -100 = 2 patterns
                                       # IInd pattern will have 
n_patterns


# Reshape input, X, to be [samples, time steps, features]
"""
Ref: https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
Three dimensions of input to LSTM are:

    Samples. One sequence is one sample. A batch is comprised of one or more samples.
    Time Steps. One time step is one point of observation in the sample.
    Features. One feature is one observation at a time step. (so one character at a time)
https://stats.stackexchange.com/questions/274478/understanding-input-shape-parameter-in-lstm-with-keras?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
"""
# 7.
X = np.array(dataX).reshape(n_patterns, lstm_seq_len, 1)
X.shape
X[0]

# 7.1 Normalize each character code
X = X / float(n_vocab)
X[0]

# 7.2 One hot encoding the output variable
y = np_utils.to_categorical(dataY)
y.shape                    # (144313,47)

#############################################################################
##################### BB. Define model ######################################
#############################################################################


# 8. Define the LSTM model
model = Sequential()

# 8.1 Note that in this case, LSTM is the first layer

model.add(LSTM(
	           256,
	           input_shape=(X.shape[1], X.shape[2])
	           )
          )
model.add(Dropout(0.2))
model.add(Dense(
	            y.shape[1],                  # 'n_vocab' will be number of neurons
	            activation='softmax'
	            )
         )

model.compile(
	         loss='categorical_crossentropy',
	         optimizer='adam'
	         )



#############################################################################
#################### CC. Load earlier model weights, if any##################
#############################################################################
# IF not, jump CC and proceed to DD.

# 9. If previouly you had run  model.fit() with checkpointing
#    and created some model-weights, you can load those
#    weights and begin iteration from there. Thus, our
#    further model development will begin 
#    This is the advantage of checkpointing
filename = "/home/ashok/Documents/13. text_generation/weights_improvement-50-1.5172.hdf5"

# 9.1
model.load_weights(filename)


#############################################################################
#################### DD. Model Checkpointing ######################################
#############################################################################
# Come here even if you loaded weights from earlier iterations in CC.


# 10.1 Because of the slowness and because of our optimization
#    requirements, we will use model checkpointing to record
#    all of the network weights to file each time an improvement 
#     in loss is observed at the end of the epoch. We will use the
#     best set of weights (lowest loss) to instantiate our 
#      generative model in the next section.
# Define the checkpoint
# Filepath can contain named formatting options,
#  which will be filled the value of epoch and keys
#   in logs (passed in on_epoch_end).
#    For example: if filepath is weights.{epoch:02d}-{loss:.2f}.hdf5,
#     then the model checkpoints will be saved with the epoch number 
#     and the validation loss in the filename.
# How to check point: See:
# Ref: https://machinelearningmastery.com/check-point-deep-learning-models-keras/

# 10.2 Filepath to store weights
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# 10.3
checkpoint = ModelCheckpoint(
	                         filepath,
	                         monitor='loss',          # Quantity to monitor
	                         verbose=1,
	                         save_best_only=True,     # Save latest best model according to the quantity monitored
	                         mode='min'               # Decision to overwrite the current save file is made based
	                                                  #  on either the maximization or the minimization of the
	                                                  #   monitored quantity. For val_loss this should be 'min'
	                         )

# 10.4 List of callback functions to apply during training.
callbacks_list = [checkpoint]                     


#############################################################################
##################### EE. Run model ######################################
#############################################################################


# 11. Fit the model
model.fit(
	      X, y,
	      epochs=20,
	      batch_size=128,
	      callbacks=callbacks_list
	      )


# 11.1 See relevant weight files
glob.glob("/home/ashok/weights-*")


#############################################################################
##################### FF. Define model ######################################
#############################################################################

#     Yes, we can also have stacks of LSTM layers
#     See http://adventuresinmachinelearning.com/keras-lstm-tutorial/


# 12. Define the LSTM model
model = Sequential()

# 12.1 Note that in this case, LSTM is the first layer
model.add(LSTM(
	           256,
	           input_shape=(X.shape[1], X.shape[2]),
	           return_sequences=True              # To allow stacking of another lstm
	                                              #  Whether to return the last output in the output
	                                              #  sequence, or the full sequence. Default False
	           )
          )

# 12.1
model.add(Dropout(0.2))

# 12.2 
model.add(LSTM(
	           256
	           )
          )

# 12.3
model.add(Dropout(0.2))

# 12.4
model.add(Dense(
	            y.shape[1],                  # 'n_vocab' will be number of neurons
	            activation='softmax'
	            )
         )

# 12.4
model.compile(
	         loss='categorical_crossentropy',
	         optimizer='adam'
	         )

# 12.5
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# 12.6
checkpoint = ModelCheckpoint(
	                         filepath,
	                         monitor='loss',          # Quantity to monitor
	                         verbose=1,
	                         save_best_only=True,     # Save latest best model according to the quantity monitored
	                         mode='min'               # Decision to overwrite the current save file is made based
	                                                  #  on either the maximization or the minimization of the
	                                                  #   monitored quantity. For val_loss this should be 'min'
	                         )

# 12.7 List of callback functions to apply during training.
callbacks_list = [checkpoint]                     

model.summary()
model.layers[2].input_shape


# 13. Fit the model
model.fit(
	      X, y,
	      epochs=20,
	      batch_size=128,
	      callbacks=callbacks_list
	      )


# 13.1 
model.summary()

# 13.2 Plot model
from skimage import io
import pydot
import matplotlib.pyplot as plt
import os
plot_model(model, to_file='shared_input_layer.png')
io.imshow('shared_input_layer.png')
plt.show()

##################### DONE ############### DONE ###########################
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



