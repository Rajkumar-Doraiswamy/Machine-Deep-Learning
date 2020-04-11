"""
Last amended: 05/05/2018
Myfolder: /home/ashok/Documents/13. text_generation
Weights folder on Google drive: http://203.122.28.230/moodle/mod/url/view.php?id=2035
Ref: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/#comment-436362

Reshape Input for Long Short-Term Memory Networks in Keras
https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

Objective:
          Text generation using small LSTM Network
		  Text Patterns from Alice in Wonderland

This file is Part II of two files (rather 3).
Files are:          
            i.   text_model_creation.py   => Develop model for text-prediction. Learn patterns
                                             of book 'Alice in Wornderland'
            ii)  text_generation.py       => Use learning to predict text, given a seed text
                                             from the book
            iii) crisp.py                 => Same as (i) but without comments for model dvt
                                             in adavance of scheduled class


Steps:
               i) Create list of unique characters, char_to_int dictionary
                  also int_to_char dictionary as before.
              ii) Create a pattern list of 100 characaters each, and store it
                  in dataX
             iii) Generate a random number and pick any pattern from dataX.
                  Call this pattern P.
              iv) Develop lstm model, as before
              v)  Load into this model, previously saved weights. So our model
                  is ready.
             vi)  Feed P into model and make a prediction of character.
             v)   Append this predicted character to our pattern, P. 
                  Create another pattern P', by shifting P, one-char to right.
             vi)  Go to (v) and repeat 1000 times   


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

# 2.1 Needed for to_categorical()
#     to convert target to one-hot-encoded form
from keras.utils import np_utils


#############################################################################
##################### AA. Read and process data file ########################
#############################################################################

"""
Even though this file is for prediction, we need to 
read and process original data file, for three purposes:

        i)  To get a seed pattern of sequence_length from the book
            from where the prediction would start. We cannot take
            any pattern from any book as the model is meant to 
            follow pattern of this book.
        ii) What are our targets? The targets are the set of unique
            characters in the book (47 in total). So we need to get
            these characters.
        iii)Our model will output integers. How do these integers
            translate to characters? We need to know this by reversing
            our earlier transformation code (of chars to intergers)
"""


# 3. Read and process Data file.
#    Load text file and covert to lowercase
filename = "/home/ashok/.keras/datasets/alice/wonderland.txt"
book_text = open(filename).read()


# 3.1 Convert text to lower case
book_text = book_text.lower()


# 3.2 Get sorted list of all unique characters in the text
unique_chars = sorted(list(set(book_text)))              # set() retains unique characters



# 3.3 Create mapping of unique_chars to integers.
#     We need this mapping to get some arbitrary
#     seed_pattern of characters from the book,
#     transformed to intergers, by this mapping.
#     This pattern will be our seed input to LSTM

char_to_int=dict()            # Empty dictionary
for i,c in enumerate(unique_chars):
	char_to_int[c]= i        # For key c set value of i



# 3.4 Summarize the loaded data
total_no_chars = len(book_text)          # 144413    len("qweqw rwer")
n_vocab = len(unique_chars)              # 47        



# 3.5 Reverse mapping: integer to character
#     We need it to transform our predictions 
#     from our model to translate to chacaters
int_to_char=dict()            # Empty dictionary
for i,c in enumerate(unique_chars):
	int_to_char[i]= c

int_to_char



# 4. Create list of patterns, dataX, as before.
#    We need to generate a seed dataset
#    of 100 characters from arbitrary position
#    LSTM will start predicting from that point.
lstm_seq_len = 100                      # Seq length of characters
dataX = []                              # Input sequence list


# 4.1 Collect all possible patterns of 100 characters in dataX
#     If there are 102 characters, then we will scan
#     2 times ie 102 - 100 = 2 ie get 2 patterns.
#     range(2) results in i = 0,1
for i in range(0, total_no_chars - lstm_seq_len, 1):   # Say i = 0
	seq_in = book_text[i:i + lstm_seq_len]             # seq_in  = book_text[0:100] 
	dataX.append([char_to_int[char] for char in seq_in])  # List comprehension



##############################################################################
##################### BB. Create model and load weights ######################
##############################################################################



# 5, Define the LSTM model as before but 
#    as we are not learning anything, so
#    no fitting and no checkpointing
model = Sequential()
model.add(LSTM(
	           256,
	           input_shape=(lstm_seq_len, 1)
	           )
          )
model.add(Dropout(0.2))
model.add(Dense(
	            n_vocab ,                 # Possible targets. Total 47.
	            activation='softmax'
	            )
         )

model.compile(
	         loss='categorical_crossentropy',
	         optimizer='adam'
	         )


# 5.1 Load network weights
#     Also available here: http://203.122.28.230/moodle/mod/url/view.php?id=2035
filename = "/home/ashok/Documents/13. text_generation/weights_improvement-50-1.5172.hdf5"

# 5.1.1
model.load_weights(filename)

# 5.1.2
model.compile(
	          loss='categorical_crossentropy',
	          optimizer='adam'
	          )



# 6. Pick a random position from within dataX
start = np.random.randint(0, len(dataX)-1)
# 6.1
seed_pattern = dataX[start]      # From within list of patterns
# 6.2
np.array(seed_pattern)           # Look at it
# 6.3
np.array([int_to_char[c]  for c in seed_pattern])





# 7. Before we get in the for-loop
#    let us traverse the loop for i=0

# BEGIN
# 7.1
i = 0
# As characters are predicted, seed_pattern will get changed
#  and each time there will be a new seed_pattern and hence
#   need for reshaping again and again to get proper
#    input, x, to model
# 7.2
x = np.reshape(seed_pattern, (1, len(seed_pattern), 1))
x = x / float(n_vocab)

# 7.3 Make prediction of next character
prediction = model.predict(x, verbose=0)

# 7.4 So what does it look like:
prediction                  # softmax output
len(prediction[0])          # 47

# 7.5
np.sum(prediction[0])       # Should sum to 1

# 7.6 At which index, we have max value
index = np.argmax(prediction)

# 7.7 This is our predicted character
result = int_to_char[index]
result


# 7.8 Write to console just the predicted character
sys.stdout.write(result)

# 7.9 Extend our seed_pattern by one character
seed_pattern.append(index)

# 7.10 Our new seed_pattern now starts from index 1 till end
seed_pattern = seed_pattern[1:len(seed_pattern)]

# Next i
# END



# 8. Generate/predict 1000 characters using LSTM
#    given a seed pattern from the book
for i in range(1000):
	# 8.1 Reshape your seed pattern as input to model
	x = np.reshape(seed_pattern, (1, len(seed_pattern), 1))
	x = x / float(n_vocab)
	# 8.2 Make prediction of next character
	prediction = model.predict(x, verbose=0)
    # 8.3
	index = np.argmax(prediction)
	# 8.4
	result = int_to_char[index]
	# 8.5
	sys.stdout.write(result)
	# 8.6
	seed_pattern.append(index)
	# 8.7
	seed_pattern = seed_pattern[1:len(seed_pattern)]


###############DONE########DONE##########################

