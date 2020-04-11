# Last amended: 19th Feb, 2018
# My Folder: /home/ashok/Documents/10.nlp_workshop/word2vec_CNN
# Ref: 
# 	https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#  How does an embedding layer work:
#    https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
#
# Objective:
#	 How to Use Word Embedding Layers for Deep Learning with Keras
#    About word embeddings and that Keras supports word embeddings via the Embedding layer.
#    How to learn a word embedding while fitting a neural network.
#    How to use a pre-trained word embedding in a neural network.

# Setting word vectors as the initial weights of embedding layer is a valid approach.
#  The word vectors will get fine tuned for the specific NLP task during training.

# We will define a small problem where we have 10 text documents,
#  each with a comment about a piece of work a student submitted.
#   Each text document is classified as positive “1” or negative “0”.
#     This is a simple sentiment analysis problem.


##################################################################
########## Example of Using Pre-Trained GloVe Embedding ##########
##################################################################

## Call libraries
# 1.0
import numpy as np
import zipfile

# 1.1 Translate each document to tokens
from keras.preprocessing.text import Tokenizer 

# 1.2 Pad each document with fixed number of tokens
from keras.preprocessing.sequence import pad_sequences

# 1.3 Keras model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding



# 2.0 Define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']

# 2.1 Define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])

# 3
### Step 1: Convert documents of words to integer-representation
#           All documents together compose one list of lists

# 3.1 Integer encode the documents
#     Prepare tokenizer
t = Tokenizer()

# 3.2 Learn all the documents
t.fit_on_texts(docs)    # docs must be a list

# 3.3 And get size of vocabulary
vocab_size = len(t.word_index) + 1

# 3.4 Integer encode the documents & look at them
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)


# 4. The sequences have different lengths and Keras prefers 
#    inputs to be vectorized and all inputs to have the same
#    length. We will pad all input sequences to have the
#    length of 4. Again, we can do this with a built in Keras
#     function, in this case the pad_sequences() function.

# 4.1 Pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs,
                            maxlen=max_length,
                            padding='post')

# 4.2
print(padded_docs)

### 5.0
### Step2:	Load word2vec file: For each word occurring
#         	in word2vec file, create a dictionary 
#         	embeddings_index, in the form: {'word' : array_of_vector_values}
#         	Consider it a transformation step of complete
#			word2vec file.

# 5.1 Load the whole glove embedding into memory
#     https://nlp.stanford.edu/projects/glove/
#     Download file: glove.6B.zip . It has four text files:
#     glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt
#     We select 50-dimension file: glove.6B.50d.txt. Data in files is, line-by-line, as:
#     to 0.68047 -0.039263 0.30186 -0.17792 0.42962 0.032246 -0.41376 0.13228 -0.29847 -0.085253 0.17118 0.22419 -0.10046 -0.43653 0.33418 0.67846 0.057204 -0.34448 -0.42785 -0.43275 0.55963 0.10032 0.18677 -0.26854 0.037334 -2.0932 0.22171 -0.39868 0.20912 -0.55725 3.8826 0.47466 -0.95658 -0.37788 0.20869 -0.32752 0.12751 0.088359 0.16351 -0.21634 -0.094375 0.018324 0.21048 -0.03088 -0.19722 0.082279 -0.09434 -0.073297 -0.064699 -0.26044
#     and 0.26818 0.14346 -0.27877 0.016257 0.11384 0.69923 -0.51332 -0.47368 -0.33075 -0.13834 0.2702 0.30938 -0.45012 -0.4127 -0.09932 0.038085 0.029749 0.10076 -0.25058 -0.51818 0.34558 0.44922 0.48791 -0.080866 -0.10121 -1.3777 -0.10866 -0.23201 0.012839 -0.46508 3.8463 0.31362 0.13643 -0.52244 0.3302 0.33707 -0.35601 0.32431 0.12041 0.3512 -0.069043 0.36885 0.25168 -0.24517 0.25381 0.1367 -0.31178 -0.6321 -0.25028 -0.38097
#
# First, a demo of what we are going to do:
# 5.2 Create an empty dictionary to store word-vectors
#     corresponding to each word in our vocabulary
embeddings_index = dict()

# 5.3 Open the word-vector file. Read directly from zipped file
file_path = '/home/ashok/.keras/datasets/glove_data/glove.6B/glove.6B.50d.txt.zip'
archive = zipfile.ZipFile(file_path, 'r')
f = archive.open('glove.6B.50d.txt')

# 5.4 Read all word-vectors in a list and also close file
all_lines = list(f)
f.close()

# 5.5 Read the first line.
#     There is a prefix 'b' to every line as strings are being read as bytes
first_line = all_lines[0]
first_line

# 5.6 Remove prefix 'b' from first_line using decode()
first_line.decode()

# 5.7 Split this line as a list of values
values = first_line.split()      # Splits at spaces
len(values)

# 5.8 Transform the whole list as an array
values = np.asarray(values)

# 5.9 Insert our first entry into the dictionary, as: key:value
embeddings_index[values[0]] = values[1:]
embeddings_index        #  Sample: {'the': array(['0.418', '0.24968', '-0.41242', '0.1217', '0.34527'...])}



# 6. Repeat all above for each of the words in the whole of vocabulary
#     
embeddings_index = dict()
for line in all_lines:
	values = line.split()
	values =[item.decode()  for item in values ]    # Remove prefix 'b' from each value
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs

# 6.1
print('Loaded %s word vectors.' % len(embeddings_index))


# 7.
### Step3: 	Create weights-matrix: 
#           From dictionary of word2vec, we pick 
#          	up only the words relevant to us &
#          	create a matrix of weights, row-wise,
#			ie, for each word in the vocabulary
#			populate a row of 50-numbers (50-dim)         
#			So our matrix size is: (vocab_len X 50)
#			This is called weight-matrix
#			Also please note that if a 'word' has
#			representation of, say, 5, its weight-vector
#			will be at 5+1 row-position ie if row-index
#			starts at 0, this weight-row will be at index: 5
#			Thus there is clear correspondence between 
#			word-representation and the row of matrix that
#			has its weight-vector

# 7.1 Initialise a weight-matrix for words in vocabulary filled-with zeros
embedding_matrix = np.zeros((vocab_size, 50))
embedding_matrix

# 7.2 First an example about how for-loop works, for word_index.items()
#     Let us see on what objects we are going to iterate
t.word_index.items()

# 7.3 Iterate now and extract 'word' as also its integer-value-representation
for word, i in t.word_index.items():
	print (word, "--",i)
	# 7.3.1 For key of 'word' in dictionary, get its integer-representation
	print(embeddings_index.get(word))


# 8.0 Next we fill embedding_matrix with weights-vector
embedding_matrix = np.zeros((vocab_size, 50))

# 8.1 Now we create the weights-matrix
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# 8.1.1 Embedding matrix creates a row-wise correspondence
		embedding_matrix[i] = embedding_vector

# 9
### Step4:	# We are now ready to define our Embedding layer as part of our neural network model.
#			To start with read this: 
#			https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
#			The Embedding has a vocabulary of 50 and an input length of 4. 
#			We will choose a small embedding space of 4 dimensions.
#			The model is a simple binary classification model. Importantly,
#			the output from the Embedding layer will be 4 vectors of 8 dimensions each, 
#			one for each word (& there are four words). 
#			We flatten this to a one 200-element vector (4 X 50) to pass on to the
#			Dense output layer.
#

# 9.1 Define model
model = Sequential()

# 9.2 Embedding layer has to be the first layer in the model
e =     Embedding(vocab_size,					# How many words in our vocabulary
                  50,							# What is the dimension of each word2vec
                  weights=[embedding_matrix],	# Initial weights-matrix
                  input_length=4,				# Max length of each document
                  trainable=False				# Should these weights be changed by training. False=> No
                  )
model.add(e)
# 9.3
model.add(Flatten())
# 9.4
model.add(Dense(1, activation='sigmoid'))
# 9.5 Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# 9.6 Summarize the model
print(model.summary())

# 10. Fit the model on our documents
model.fit(padded_docs,
          labels,
          epochs=50,
          verbose=0
          )

# 10.1 Evaluate the model
loss, accuracy = model.evaluate(
	                            padded_docs,
	                            labels,
	                            verbose=0
	                            )
print('Accuracy: %f' % (accuracy*100))

##########################################################################

# What does the Embedding layer do?
# Ref: https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding

model = Sequential()
model.add(Embedding(5, 2, input_length=5))

input_array = np.random.randint(5, size=(1, 5))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
##################################################################