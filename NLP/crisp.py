# Last amended: 30th April, 2018
# Myfolder: /home/ashok/Documents/13. text_generation
# Weights folder on Google drive: http://203.122.28.230/moodle/mod/url/view.php?id=2035
# Ref: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/#comment-436362

# WHY THIS FILE? -------------------?????????
# This file has same as that of 'text_model_creation.py'
# But it is without comments. This file can be 
# distributed to students in adavance of class to generate model
# weights to be used during class.


#***************************************************************************
# Some variable names may be different than those in file: text_model_creation.py
#***************************************************************************

import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils



filename = "/home/ashok/Documents/13. text_generation/alice/wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text


unique_chars = sorted(list(set(raw_text)))              
unique_chars 							    			



np.array(unique_chars)     
len(np.array(unique_chars))


char_to_int=dict()
for i,c in enumerate(unique_chars):
	char_to_int[c]= i 





n_chars = len(raw_text) 
n_vocab = len(unique_chars)



seq_length = 100
dataX = []      
dataY = []
s = ""

for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length] 
	s = seq_in
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])


dataX[0] 
n_patterns = len(dataX)
n_patterns


X = np.array(dataX).reshape(n_patterns, seq_length, 1)
X[0]
X.shape


X = X / float(n_vocab)
X[0]

y = np_utils.to_categorical(dataY)

# Define the LSTM model
model = Sequential()
model.add(LSTM(
	           256,
	           input_shape=(X.shape[1], X.shape[2])
	           )
          )
model.add(Dropout(0.2))
model.add(Dense(
	            y.shape[1],
	            activation='softmax'
	            )
         )

model.compile(
	         loss='categorical_crossentropy',
	         optimizer='adam'
	         )



filepath="/home/ashok/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
	                         filepath,
	                         monitor='loss',          
	                         verbose=1,
	                         save_best_only=True,     
	                         mode='min'               
	                         )


callbacks_list = [checkpoint]                     

model.fit(
	      X, y,
	      epochs=25,
	      batch_size=128,
	      callbacks=callbacks_list
	      )



