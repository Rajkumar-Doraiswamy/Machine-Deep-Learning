"""
Last amended: 20th Feb, 2018
My folder: /home/ashok/Documents/10.nlp_workshop

Ref:
https://github.com/hundredblocks/concrete_NLP_tutorial
https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb

DataSource: Disasters on social media
  	Contributors looked at over 10,000 tweets retrieved with
  	a variety of searches like “ablaze”, “quarantine”, and
  	“pandemonium”, then noted whether the tweet referred to
  	a disaster event (as opposed to used in a joke or with 
  	the word or a movie review or something non-disastrous). 

Objective:
          Try to correctly predict tweets that are
          about disasters using word2vec


	$ source activate tensorflow
	$ ipython             

"""


## 1.0 Clear memory and call libraries
%reset -f

# 1.1 Array and data manipulation libraries
import pandas as pd
import numpy as np

# 1.2 nltk is a leading platform for building Python programs to work
#      with human language data
#from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
# Pads all sequences to same specified length
from keras.preprocessing.sequence import pad_sequences
# Convert a class vector (integers) to binary class matrix.
from keras.utils import to_categorical


# 1.3
# 1.3 Keras model
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Dropout, Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model


# 1.4 Accuracy reports/metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 1.5 For word2vec manipulations
import gensim

# 1.6 Utilities
import os,time

########################### Data reading, cleaning and counting ###########################

# Define come constants
EMBEDDING_DIM = 300			 # Google's word2vec dimension (standard)
MAX_SEQUENCE_LENGTH = 35     # Each social-media comment should be stretched to this size by padding, if required
VALIDATION_SPLIT=.2          # train-test split


## 2.0 Sanitizing inuts
#  2.1 Where are my data & other files?
os.chdir("/home/ashok/Documents/10.nlp_workshop/word2vec")


# 2.2 Let's inspect the data
#     See StackOverflow: https://stackoverflow.com/a/18172249
#     Default 'header' is 'infer'
questions = pd.read_csv("socialmedia_relevant_cols.csv",
                        encoding="ISO-8859-1"      # 'utf-8' gives error, hence the choice
                        )

# 2.3 Nevertheless we may assign column names
questions.columns=['text', 'choose_one', 'class_label']
# 2.4 Inspect the data now
questions.head()
questions.tail()
questions.describe()


# 2.5 Clean the data
#     Let's use a few regular expressions to clean up data,
#     and save it back to disk for future use
# Ref: http://www.regexlib.com/(X(1)A(5VYROh19ihtNP-JfoQjDSiymxutkQHoWK2wjkPydrzYhq5c450yDq5RxRfvQsrh09BHiCyJCNwIztpf_CsfLdb_1KDUgs1fy1BApFZ3xE8T_lQnsDlHhP1OEtO4T58M2x4_sJkRwe2gdtYRimfwin6FDQLaahvTmhUwWQ8sBhDg09H7kKKUevqzC8N1j4LNT0))/cheatsheet.htm?AspxAutoDetectCookieSupport=1
#     Experiment like this:
#
#          "@abc".replace(r"@", "at")
#
def standardize_text(df, text_field):
    # 2.5.1 Matches http followed by one or more non-white-space (\S)characters
    #       ie. include all alphabetical and numeric characters, punctuation, etc.
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
   	# 2.5.2 Matches http. Note the sequence of these instructions
    df[text_field] = df[text_field].str.replace(r"http", "")
    # 2.5.3 Matches @ followed by one or more non-white-space characters
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    # 2.5.4 Matches any character not included within square brackets
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

# 2.6 Apply function and clean now
questions = standardize_text(questions, "text")

# 2.7 Save cleaned data to a file for future
questions.to_csv("clean_data.csv")
questions.head()



# 2.8 Read from saved file
del questions
clean_questions = pd.read_csv("clean_data.csv")
clean_questions.tail()

# 2.9 Let's look at our class balance.
clean_questions.groupby("class_label").count()



########################### Some Exprimentation ###########################
##                          Only three documents     				##

###### 4. Examining word2vec
# Every 'word' is projected into a multi-dimensional space. 
# Load 300-dimensioned slimmed-down version of actual Google model 
#  Actual model's file size is many GBs (word-dimensions are 300)
#   Actual Google's model has words from other languages such as Chinese

# 4.1 Where is my word2vec file
word2vec_path = "/home/ashok/.keras/datasets/google_word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz"

# 4.2 Load model into memory
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


# 5. So what is the vector for 'dog' and its length
word2vec['dog']    		# array([ 1.71980578e-02, -7.49343913e-03, -5.79820201e-02, ...]
len(word2vec['dog'])	# 300

# 5.1 And for 'cat'
word2vec['cat']

# 5.2 And for 'a'
word2vec['a']			# Gives error

# 5.3 Check if a word is in word2vec
'dog' in word2vec        # True
'a'   in word2vec        # False
',' in word2vec          # False


##### 6 Using keras-tokenizer to prepare weights-matrix ######
#####   Its utility comes from its methods especially: texts_to_sequences() ####


#6.1	Two Tokenizers:	There are two kinds of tokenizers
#		One is nltk-tokenizer and the other is keras-tokenizer.
#       Both work with tokens differently. nltk-tokenizer
#       transforms ['dog and cat'] to ['dog','and','cat']
#       while keras tokenizer will map each word to a unique
#       integer number, as: 
#		{'dog': 2, 'and': 1, 'cat': 3} ; min integer is 1.


# 6.1	Let our list of (three) documents be: 
#       We will be creating a weights-matrix for 
# 		these documents

docs = ['Well done!',              # unique words = 5
		'Good good work',
		'nice work']

# 6.2	Examining keras tokenizer
#		We instantiate keras tokenizer with 
#		vocabulary size (NOT vocabulary)

keras_tokenizer = Tokenizer(num_words= 5)   # Restrict maximum number of words to top most frequent 'num_words'

# 6.3	Fit the tokenizer on the three documents
keras_tokenizer.fit_on_texts(docs)


# 6.4	Let us see results from this fitting
#		tokenizer knows lots of secrets
keras_tokenizer.word_counts			# Each word appears how many times?
keras_tokenizer.document_count		# How many dcuments? 5 
keras_tokenizer.word_docs			# Each word appears in how many documents?
keras_tokenizer.word_index			# Dictionary of words and their uniquely assigned integers.
# 6.5 Integer representation of words?
keras_tokenizer.word_index['done']	# 4
keras_tokenizer.word_index['well']	# 3
keras_tokenizer.word_index['good']	# 1
keras_tokenizer.word_index['work']	# 2

# 6.6 Get a sequence of integers 
keras_tokenizer.texts_to_sequences(docs)    #  [[3, 4], [1, 1, 2], [2]]

# 6.7
word_index = keras_tokenizer.word_index		#  {'done': 4, 'good': 1, 'nice': 5, 'well': 3, 'work': 2}
word_index.items()              			#  dict_items([('work', 2), ('nice', 5), ('well', 3), ('good', 1), ('done', 4)])
len(word_index)								# 5	


# 6.8 Check if a word exits in word2vec, get its array
#      else get an array of 300 zeros
our_word = 'good'
our_vector = word2vec[our_word] if our_word in word2vec else np.random.rand(EMBEDDING_DIM)
our_vector.shape

# 6.9 Initialise a weights matrix of size (len(word_index)+1, EMBEDDING_DIM)
#     len(word_index)+1 because minimum word_index starts from 1 and NOT 0
embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
embedding_weights.shape


# 6.10 Fill the zero-matrix with actual weight-vectors 
for word,index in word_index.items():
    embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

# 6.11 Look at the embedding_weights
embedding_weights.shape			# 6 X 300
embedding_weights				# 0th index is all zero


########################### Exprimentation Finished  ###########################
########################### Working with actual data ###########################

########################### Prepare weight-vector-matrix ###########################


## 7.   Using keras tokenizer
#		Now with the actual text
#       Convert whole column to list of lists 
text_list = clean_questions["text"].tolist()
len(text_list)		# 10876
text_list[0]		# 'just happened a terrible car crash'


# 8. Instantiate a keras-tokenizer and generate sequences vector
del keras_tokenizer      # Delete the earlier object

# 8.1
keras_tokenizer = Tokenizer(num_words=None)   # None => Consider all words in our set
keras_tokenizer.fit_on_texts(text_list)
# 8.2
sequences = keras_tokenizer.texts_to_sequences(text_list)
type(sequences)		# list
len(sequences)		# 10876
sequences[0]		# [28, 776, 2, 1506, 126, 95]


# 8.3 word_index: Which word maps to which integer
word_index = keras_tokenizer.word_index

# 8.4 How many unique words?
len(keras_tokenizer.word_index)		# 19097 unique words

# 8.5 About np.zeros
np.zeros((3,2))

# 8.6 Create a zero-filled-array of shape: number_of_words X 300
#     len(word_index)+1 because minimum word_index starts from 1 and NOT 0
embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
embedding_weights.shape


# 8.7 For every word and its index in each item
for word,index in word_index.items():
    embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)


embedding_weights.shape			# (19098,300)


########################### Prepare token sequences & split data ###########################


# 9. Make all individualinteger sequences of same length
cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
cnn_data.shape				# (10876, 35)
cnn_data[0]   				# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,776,2,1506,126,95]

# 9.1 Convert integer labels to one-hot-coded form
to_categorical([0,0,1,2,2])
labels = to_categorical(np.asarray(clean_questions["class_label"]))
type(labels)
labels[:2]         # Look at first two labels

# 10. Split dataset now. First random shuffling
indices = np.arange(cnn_data.shape[0])
indices

# 10.1
np.random.shuffle(indices)
indices

# 10.2 Fancy indexing
cnn_data = cnn_data[indices]
labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])    # 2175


# 10.3 Train/validation data
x_train = cnn_data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = cnn_data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


########################### Modeling ###########################
## Simple model first

# 11 Define model
model = Sequential()

# 11.1 Embedding layer has to be the first layer in the model
e =     Embedding(len(word_index)+1,			# How many words in our vocabulary
                  300,							# What is the dimension of each word2vec
                  weights=[embedding_weights],	# Initial weights-matrix
                  input_length=35,				# Max length of each document
                  trainable=False				# TRY BOTH WITH False and True
                  )
# 11.2
model.add(e)
# 11.3
model.add(Flatten())
# 11.4
model.add(Dense(3,								# Target values are of 3 types
                activation='sigmoid'
               )
         )
# 11.5 Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',		# Not binary
              metrics=['acc']
              )

# 11.6 Summarize the model
print(model.summary())

# 12. Fit the model on our documents
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=10,
          batch_size=128
          )


############################################3
## Complex model next
## Based on Yoon Kim model (https://arxiv.org/abs/1408.5882)

# 13. Define a Convolution network
#     
def ConvNet(wts_matrix, max_sequence_length, vocab_words, embedding_dim, unique_target_labels, isEmbeddingLayerTrainable=False, extra_conv=True):
    # Ref: https://github.com/keras-team/keras/issues/853
    embedding_layer = Embedding(vocab_words,             			# Size of the vocabulary, i.e. maximum integer index + 1
                            embedding_dim,             				# EMBEDDING_DIM, 300
                            weights=[wts_matrix],  					# Initial weights from word2vec
                            input_length=max_sequence_length,       # MAX_SEQUENCE_LENGTH; 35
                            trainable=isEmbeddingLayerTrainable		# Expt both with False and True
                            )

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)

    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv==True:
        x = Dropout(0.5)(l_merge)  
    else:
        # Original Yoon Kim model
        x = Dropout(0.5)(pool)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    #x = Dropout(0.5)(x)

    preds = Dense(unique_target_labels, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',							# Not binary
                  optimizer='adam',
                  metrics=['acc'])

    return model



# 13.1 Define/Instantiate model
model = ConvNet(embedding_weights,											# weights-matrix
                MAX_SEQUENCE_LENGTH,										# 35
                len(word_index)+1,											# 19098
                EMBEDDING_DIM, 												# 300
                len(list(clean_questions["class_label"].unique())),			# 3
                False      													# trainable is False
                )

# 13.2
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=3,
          batch_size=128
          )


######################################################################
