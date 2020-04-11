"""
Last amended: 28th June, 2019
My folder: /home/ashok/Documents/8.rnn
Ref: Page

Objectives:
        i)   To use SimpleRNN for Sentiment analysis
        ii)  To understand structure of Embedding layer
	    iii) To perform tokenization, see file:
             8.rnn/3.keras_tokenizer_class.py OR file
             8.rnn/0.document_to_id_conversion.py
	         And a quick note at the end of this code.

"""

# 1.0 Call libraries
%reset -f
import numpy as np

# 1.1 Import module imdb & other keras modules
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

# 1.2 Misc
import matplotlib.pyplot as plt
import time


# 2.1 Define some constants
max_vocabulary = 10000        # words
max_len_review = 500          # words

# 2.2 About imdb module
help(imdb)

# 2.3 Get imdb reviews. Limit vocabulary to size max_vocabulary
#      imdb reviews will be downloaded unless available at ~/.keras/datasets
# ************
#      See comments at the end as to how to quickly convert text to integers
# ************
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_vocabulary)

# 2.4 The following will download dictionary
#       of word indices back to words
#        Do not attempt
# imdb.get_word_index()

# 2.5 About data
x_train.shape      # (25000,)
x_test.shape       # (25000,)
y_train.shape      # (25000,)
y_test.shape       # (25000,)

# 2.5.1
x_train[:2]       # Have a look at two elements
y_train[:4]       # array([1, 0, 0, 1])

len(x_train[1])

# 2.6 Check max and min length of reviews
maxLen = 0         # Start with a low number
minLen = 200       # Start with a high number
for i in range(x_train.shape[0]):
    if maxLen < len(x_train[i]):
        maxLen = len(x_train[i])
    if minLen > len(x_train[i]):
        minLen = len(x_train[i])

# 2.6.1
maxLen         # 2494
minLen         # 11


# 2.7 We want to pad all sequences to max_len_review size.
#     Reviews more in size will be truncated and less in
#     size will be padded with zeros
help(sequence.pad_sequences)

# 2.7.1
x_train = sequence.pad_sequences(
                                 x_train,
                                 maxlen = max_len_review,
                                 padding = 'pre'
                                 )


type(x_train)          # numpy.ndarray
x_train.shape          # (25000, 500) Each sequence becomes one row

x_train[:2,:3]
x_train[24996:, 496:]

# 3.0 Model now
model = Sequential()
# 3.1 Embedding layer
model.add(Embedding(max_vocabulary,            # Decides number of input neurons
                    32,                        # Decides number of neurons in hidden layer
                    input_length= max_len_review) # (optional) Decides how many times
                                                  # RNN should loop around
                                                  # If omitted, decided autoamtically
                                                  # during 'model.fit()' by considering
                                                  # x_train.shape[1]
                    )

# 3.2
# It is instructive to see number of parameters
#  in the summary. This tells us about the Embedding
#   layer as being two layered network with no of neurons
#    as max_vocabulary and output (hidden) layer with 32 neurons
model.summary()

# 3.3 Ideally we should be adding not one RNN but as many RNNs as
#     there are timesteps ie sequence length or 'max_len_review'.
#     But we add just one and perform internal looping. Note that
#     internal weights and hence LSTM parameters remain same from one
#     'timestep' to another 'timestep'. You can verify this by
#     changing the value of max_len_review and seein that number
#     of parameters in the model summary after adding the following
#     do not change.

model.add(SimpleRNN(32,
                    return_sequences = False   # Make it True
                                               # And add layer #3.4
                    )
                    )   # Output


# 3.4 JUMP FOLLOWING UNLESS YOU WANT 'RNN' ABOVE 'RNN'. IT WORKS.
#     BUT TAKES TIME.
# 3.4 Make return_sequences = True in 3.3 above, before you add
#     the following layer with return_sequences = False. Else JUMP it.
#     ACCURACY IS SOMEWHAT MORE

model.add(SimpleRNN(
                    32,
                    return_sequences = False   # Make return_sequences = True
                                               # in earlier RNN for this to work
                    )
                    )   # Output



"""
Why SimpleRNN adds 2080 parameters?
    input_features * output_features = 32 * 32  = 1024
    state_t * output_features        = 32 * 32  = 1024
    Bias                                            32
    Total                                         2080
This total is INDEPENDENT of sequence length or timesteps.
"""
model.summary()     # Why SimpleRNN adds 2080 parameters?
                    # input_features * output_features = 32 * 32  = 1024
                    # state_t * output_features        = 32 * 32  = 1024
                    # Bias                                            32
                    # Total                                         2080

# 3.3
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
help(model.compile)

# 3.4
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['acc'])



help(model.fit)

# 4.0
epochs = 10
start = time.time()
history = model.fit(x_train,
                    y_train,
                    batch_size = 32,             # Number of samples per gradient update
                    validation_split = 0.2,      # Fraction of training data to be used as validation data
                    epochs = epochs,
                    shuffle = True,              # Shuffle training data before each epoch
                    verbose =1
                    )
end = time.time()
(end-start)/60


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

# 6.0
plot_learning_curve()

########################### I am done ###########################

###############################################################
# Here is Quick text to integer conversion
#  For more study, please see file: 3.keras_tokenizer_class.py
###############################################################

from keras.preprocessing.text import Tokenizer
texts = ["Sun shines brightly  in June!",
         "Star light shines on water?",
         "Water is flowing.",
         "Flowing water, shines",
         "Sun is star?",
         "World shines",
         "Star also shines",
         "water is life",
         "Sun is energy"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
tokenizer.word_index       # Index is created based on word-frequencies
                           # Most frequent word gets the least index
tokenizer.texts_to_sequences(texts)
#########
