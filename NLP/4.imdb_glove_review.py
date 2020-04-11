# Last amended: 16th Dec, 2018
# My folder: /home/ashok/Documents/10.nlp_workshop/imdb
# VM: lubuntu_deeplearning
# Ref: Page 188, Chapter6, Deep Learning with Python, Francois Chollet

# Objectives:1. Using predefined word_embeddings
#               to classify sentiments in imdb dataset
#            2. Using Keras Embedding layer
#            3. Using Glove word2vec


##  NOTE: Use theano and NOT tensorflow
##        i)  configure file .keras/keras.json
##       ii)  Deactivate tensorflow
##       iii) Activate theano
##       iv)  export "MKL_THREADING_LAYER=GNU"

"""
Steps:

AA>
1. Call libraries to tokenize-text, pad_sequences,
   and model building libraries

2. Define Constants: Decide upon max-comment length, no of training samples,
                     no of validation samples and
                     no of top-frequetly-occurring-words

3. Read all comments as:
    ['comment1', 'comment2', 'comment3']
   And read all labels as per folder name
    [0,1,0]

4. Tokenize all text to sequences
      sequences: [[2,3], [4,5,6]]
5. Pad sequences:
      [[0,2,3], [4,5,6]]
6. Shuffle sequences:
      [[4,5,6], [0,2,3]]
7. Split list of sequnces as per no of training samples/validation samples

BB>
### Glove data
8. Read glove vector-space model into a dictionary as:
    {'the':np.array([1.1,2.2]), 'for' : np.array([2.3, 3.4]) }
9. Create an embedding matrix for top-frequetly-occurring-words
    and in that order, as:

				1.1   2.2
                2.3   3.4
                6.7   7.8
CC>
## Model Bulding:
10. Build Sequential model
	1. Embedding layer + Classification layer
    2. Set weights in embedding layers as per Glove matrix
    3. Freeze embedding layers from any futher trainning
    4. Add classification layer
    5. Compile and build the model

11. Train the model to classify, given vector-space-model of text

DD>
12. What is embedding layer? See detailed explanation at the end.

FF>
13. Accuracy is not that good because:
    1. word2vec is just of size 50 and not 300
    2. Traning samples taken are very less, just 400
    3. Comment length is taken to be 100
    4. Vocabulary size is limited to 10000 words
    5. Classifier has no dropouts


"""


# 1.0 Call libraries
%reset -f
import os
import numpy as np

# 1.1 Library to tokenize the text (convert to integers)
from keras.preprocessing.text import Tokenizer
# 1.2 Make all sentence-tokens of equal length
from keras.preprocessing.sequence import pad_sequences
# 1.3 Modeling layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

# 2.0 Constants and files processing
# This dataset contains just 1000 comments each per neg/pos folder
imdb_dir = '/home/ashok/Documents/10.nlp_workshop/imdb/'
train_dir =os.path.join(imdb_dir, 'train')
train_dir


# 2.1 Look into folders
os.listdir(train_dir)                # ['pos', 'neg']
os.listdir(train_dir + '/pos')       # List of files
os.listdir(train_dir+'/neg')         # List of files


# 2.2 How many files are there?
fname = train_dir+'/neg'
len(os.listdir(fname))               # 1000

fname = train_dir+'/pos'
len(os.listdir(fname))  # 1000


# 2.3
maxlen_comment = 100          # If comment exceeds this, it will be truncated
training_samples = 400        # Use just 400 comments as training data
validation_samples = 500      # Use 500 samples for validation
max_words = 10000             # Select top 10000 words



# 2.3 List of sentiment labels and comments
#     Start with none
labels = []
texts = []

# 2.4 Read files in each folder.
#     As we do so, we  append each
#     comment in 'texts' list as a string
#     and also append its sentiment in
#     labels
for label_type in ['neg', 'pos']:
    # 2.3.1 Which directory
    dir_name = os.path.join(train_dir, label_type)
    # 2.3.2 For every file in this folder
    for fname in os.listdir(dir_name):
        # 2.3.3 Open the file
        f = open(os.path.join(dir_name, fname))
        # 2.3.4 Append its text to texts[]
        texts.append(f.read())
        f.close()
        # 2.3.5 And if the directory was 'neg'
        if label_type == 'neg':
            # 2.3.6 All comments are negative
            labels.append(0)
        else:
            labels.append(1)

# 2.4 What have we got?
len(texts)       # 2000 comments
texts[0]         # Read one comment
texts[1]         # Read another

labels[:5]       # look at labels

# 3. Start processing

# 3.1 Create object to tokenize comments. Pick up
#     top 'max_words' tokens (by frequency of occurrence)
#      Its full syntax is:
#   Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~
#      ', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
tokenizer = Tokenizer(num_words = max_words)

# 3.2 Learn the complete comments
tokenizer.fit_on_texts(texts)

# 3.3 Tokenize texts. Actual transformation occurs here
sequences = tokenizer.texts_to_sequences(texts)

# 3.3.1
type(sequences)     # It is a list
sequences[:2]
# 3.4 Have a look
len(sequences)          # How many of them: 2000
sequences[0]            # This is the first comment
texts[0]                # Compare it with actual text
                        #  Note that most frequent words have been
                        #  numbered less, as 'the', 'of'

# 3.5 Which one of the comments have less than 100 words
for i in np.arange(len(sequences)):
    l = len(sequences[i])
    if (l < 50):
        print(i)     # 45

len(sequences[45])   # 16

# 3.6 Get word-to-integer dictionary?
word_index = tokenizer.word_index
word_index

# 3.7 Which are the top few most frequent words
for i in word_index:
    if word_index[i] < 10:
        print(i)

# 3.8
len(word_index )            # 43296

# 3.9 Make all sequences of equal length
data = pad_sequences(sequences, maxlen = maxlen_comment)
data[45]         # Check
len(data[45])    # 100
data.shape       # (2000 X 100)


# 3.91 And what about labels?
type(labels)      # list
labels = np.asarray(labels)   # Transform to array
labels


# 4.0 Shuffle comments randomly
# 4.0.1 First generate a simple sequence
indices = np.arange(data.shape[0])

#4.0.2 Shuffle this sequence
np.random.shuffle(indices)
indices

# 4.1 Extract data and corresponding labels
data = data[indices, ]
labels = labels[indices]

# 4.2 Prepare train and validation data
X_train = data[:training_samples, ]
y_train = labels[:training_samples]

x_val = data[training_samples:training_samples+validation_samples,]
y_val = labels[training_samples:training_samples+validation_samples]

# 5. Where is glove?
glove_dir = '/home/ashok/.keras/datasets/glove_data/glove.6B'


# 5.1 Put all glove vectors in a dictionary
embeddings_index = {}

# 5.2 'glove.6B.50d.txt' is a text file
#      Rename it as: glove_6B_50d.txt
#         cd /home/ashok/.keras/datasets/glove_data/glove.6B
#         unzip glove.6B.50d.txt.zip
#         mv glove.6B.50d.txt glove_6B_50d.txt
#     You can read few vectors using 'cat'
#        cd /home/ashok/.keras/datasets/glove_data/glove.6B
#        cat glove_6B_50d.txt | more
#    OR  cat glove_6B_50d.txt | more

# 5.2 Start reading the file line by line
f = open(os.path.join(glove_dir, 'glove_6B_50d.txt'), 'r')
for line in f:
    # 5.2.1 Split each line on ' '
    values = line.split()
    # 5.2. The first token is the word
    word = values[0]
    # 5.2.3 Rest all numbers in the line are vectors for this word
    coefs = np.asarray(values[1:], dtype = 'float32')
    # 5.2.4 Update embeddings dictionary
    embeddings_index[word] = coefs

f.close()

# 5.3 Have a look at few vectors
embeddings_index['the']
len(embeddings_index['the'])            # 50
len(embeddings_index)                   # 400000

# 6. We need to transform this dictionary in max_words X embedding_dim
#    OR, as: 10000 X 50 matrix. In this matrix 1st row is a vector for
#    1st word, IInd row for IInd word and so on. The sequence of words
#    is as in word_index:

embedding_dim = 50
# 6.1 Get an all zero matrix if equal dimension
embedding_matrix = np.zeros(shape = (max_words, embedding_dim))
embedding_matrix.shape


# 7. Now fill embedding matrix with vectors
# word_index.items() is a tuple of form ('the', )

type(word_index)                # Dictionary
list(word_index.items())[0:5]   # key-value tuple

# 8 For every word and token in every tuple of word_index.items()
for word, i in word_index.items():
    # 8.1 If token value is less than 10000
    #     Token coding/numbering is in order of frequency of word
    #     That is ('the', 5) occurs more than ('you', 6)
    if i < max_words:
        # 8.2 For the particluar key (ie 'word') get value-vector
        embedding_vector = embeddings_index.get(word)
        # 8.3 Store the vector in the matrix
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# 9. Have a look
embedding_matrix
embedding_matrix.shape    # 10000 X 50

# 11. Finally develop the model
model = Sequential()
#                   10000        50                           100
model.add(Embedding(max_words,embedding_dim, input_length = maxlen_comment))
model.add(Flatten())

#  11.1 Flattened neuron size: (max_words,embedding_dim  X maxlen_comment)
model.summary()
# 11.2
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.summary()

# 12. Seed embedding layers with glove weights (10000 X 50)
model.layers[0].set_weights([embedding_matrix])
# 12.1 And let weghts in this layer not change with back-propagation
model.layers[0].trainable = False

# 13. compile the model
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

# 14. Train the model for 30 epochs
#     With just 400 training samples we get 59% validation accuracy
history = model.fit(X_train,y_train,
                    epochs = 100,
                    batch_size = 32,
                    validation_data = (x_val,y_val)
                    )
###################################################################################
"""
How embedding layer works?
==========================

Ref: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

Here is how one can think of embedding layer working:
Embedding layer consists of just input layer and hidden layer And
no output layer.

Given an input of size 1 X 5 and weight matrix of size, 5 X 3,
output of hidden layer are three values.

          (1X5) matmult (5X3) = (1 X 3)

These values represent the weight vector of input word.

Input:
   We input one one-hot-encoded word ([0 0 0 1 0])to embedding
   layer here. Thus, our word- vocabulary size is 5 (length of
   one-hot-encoded word) and our word-vector size is 3 (no of
   neurons in the hidden layer):

  One-hot                  Layer with    word2vec
  encoded     input        linear        output
  word        layer        activation
    0   ->     +
    0   ->     +            +       ->   10
    0   ->     +            +       ->   12
    1   ->     A            +       ->   19
    0   ->     +
              |               |
              |<- Embedding-->|
              |    layer      |

Explanation:
   Consider just this one-hot-encoded input: [0 0 0 1 0]
   Here weights from neuron 'A' reaching to three
   neurons of hidden layer are from top-to-bottom
   10,12 and 19. Other (weights * signals from
   other neurons) also reach these three
   hidden-layer neurons to get summed up but as
   input signal in each of the other cases is zero,
   output signals of hidden layers remain
   10 *1,12*1 and 19*1 respectively.
   Note that activation function is linear. Thus
   embedding layer, in effect, outputs just the
   weight vector of a given one-hot-encoded word.


Example with all weights shown:
-------------------------------

    0   ->     +   17 24  1
    0   ->     +   23  5  7     +       ->   10
    0   ->     +    4  6 13     +       ->   12
    1   ->     A   10 12 19     +       ->   19
    0   ->     +   11 18 25


Matrix Multiply input signal with weight matrix
and see what you get:
                           17   24  1
                           23    5  7
                            4    6  13
[0   0   0   1   0]    X   10   12  19    =   [10   12  19]
                           11   18  25


So if our sentence consists of three words, as
each word of sentence is fed, a weight vector is
output. We may have for some sentence, a 2D feature
array as:

               11 18 25
               4   6 13
               17 24  1

All sentences mus be of equal length. So, another
three-word sentence may give us a 2D array as:

                10 12 19
                11 18 25
                4   6 13

After the embedding layer, each one of these 2D inputs
is first flattened and then analysed.
(In this case we will have 3 X 3 = 9 neurons of flattened layer)


'terrible' vs 'awful'
--------------------
If most similar word to 'terrible' is
'awful', it means that the skip-gram vector
for both the words would also be the same. 
Reason is this. Whenever 'terrible' is
used, it has some context words. The
word 'awful' will also be occurring with
the same conext words.
So both words must have the same word2vec.



# Last amended: 21/12/2018

Neural network
Dot product of Inputs and weights
=================================

Neural network and weights

                      n1                      
   x1->   1           n2
   x2->   2           n3
                      n4

If weight and inputs are written like as follows:
              
W=weights arriving at neurons=>    n1    n2    n3    n4              
  Write as: =>                     w1a   w2a   w3a   w4a  
                                   w2b   w2b   w3b   w4b
X= Inputs: [ x1   x2]

Then:
    X dot W   = [w1a * x1 + w2b * x2   w2a * x1 + w2b * x2  w3a * x1 + w3b * x2 w4a * x1 + w4b * x2]
====================================


Download GloVe from:
  https://nlp.stanford.edu/projects/glove/

Download of Google wrod2vec model based on Google news
  https://github.com/mmihaltz/word2vec-GoogleNews-vectors

"""
