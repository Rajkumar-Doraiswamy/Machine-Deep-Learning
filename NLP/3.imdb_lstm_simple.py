# Last amended: 20/12/2018
# Myfolder:
# Ref: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#      https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e

# Objective: Understanding how Word Embedding Layer of Keras + LSTM work
#            Train embedding layer to transform text to vectors using LSTM

"""
Steps:
       a. Call librariesR
       b. Read files and labels
       c. Clean comments
       d. Convert text to integer sequences
       e. Split data into train and test
       f. Build & run LSTM model now
       g. Check accuracy



How does an embedding layer work?
      https://stats.stackexchange.com/a/305032
      https://stats.stackexchange.com/a/325011

 Is embedding layer in Keras is doing the same as word2vec. No
 Remember that word2vec refers to a very specific network setup
 which tries to learn an embedding which captures the semantics
 of words ie words that appear in context. With Keras's embedding
 layer, you are just trying to minimize the loss function, so if
 for instance you are working with a sentiment classification problem,
 the learned embedding will probably not capture complete word semantics
 but just their emotional polarity...

 Thus while weights learnt in word2vec may work in any classifier,
 weights learnt by an embedding layer may work in the context
 of the subject on which it is being trained on.

"""


################## AA. Call libraries


# 1.0 Call libraries
%reset -f
import numpy as np

# 1.1 Keras text and sequence processing
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

# 1.2 Keras modeling layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM
from keras.utils import to_categorical
import os



################## BB. Read files and labels

# 2.0 Constants and files processing
#     This dataset contains just 1000 comments each per neg/pos folder
imdb_dir = '/home/ashok/Documents/10.nlp_workshop/imdb/'
train_dir =imdb_dir + 'train'
train_dir             # '/home/ashok/Documents/10.nlp_workshop/imdb/train'


# 2.1 Look into folders
os.listdir(train_dir)                # ['pos', 'neg']
neg_dir = train_dir+'/neg/'
pos_dir = train_dir+'/pos/'

# 2.1.1 Get list of files in each
os.listdir(pos_dir)         # List of files
os.listdir(neg_dir)         # List of files


# 2.2 How many files are there?
len(os.listdir(neg_dir))               # 1000
len(os.listdir(pos_dir))  # 1000


# 2.3 Define some constants
max_sentence_length = 200      # If comment exceeds this, it will be truncated
training_samples = 1500        # Use just 1500 comments as training data
validation_samples = 500       # Use 500 samples for validation
vocab_size = 20000             # Select top 20000 words



# 2.4 Start reading each directory, one by one

# 2.4.1  Start with empty lists
labels = []
texts = []

# 2.5 For each file iin the directory
#     open it, read it and append to 'texts'
files = os.listdir(neg_dir)
for file in files:
    # 2.5.1 Open the file
    f = open(neg_dir+file)
    # 2.5.2 Append its text to texts[]
    texts.append(f.read())
    # 2.5.3 Close this file
    f.close()
    # 2.5.4 Being 'neg' folder, let label be 0
    labels.append(0)

# 2.6 Read data from files in 'pos' folder
files = os.listdir(pos_dir)
for file in files:
    f = open(pos_dir+file)
    texts.append(f.read())
    f.close()
    labels.append(1)

# 2.7 What have we got?
len(texts)       # 2000 comments
texts[0]         # Read one comment
texts[1]         # Read another
labels[:5]       # look at labels

################## CC. Clean comments

# 3.1 Clean texts
#     Simple cleaning
st = ""
for doc in texts:
    doc = doc.lower()
    doc = doc.replace('!', "")
    doc = doc.replace('.', "")
    doc = doc.replace('[', " ")
    doc = doc.replace(']', " ")
    doc = doc.replace('(', " ")
    doc = doc.replace(')', " ")
    doc = doc.replace('@', " ")
    doc = doc.replace('&', " ")
    doc = doc.replace(',', " ")
    doc = doc.replace('?', "")
    st = st + " " + doc

# 3.2
type(st)
len(st)                # 7716644
st[0:1000]              # Just some values
len(st.split(' '))            # Total words: 1598253
len(set(st.split(' ')))       # Unique 56119


# 3.3 Define array of sentiment class labels for documents
#     labels = np.array([0,0,0,0,0,1,1,1,1,1])
labels = np.asarray(labels)

################## DD. Convert text to integer sequences


# 4. Transform all documents to integer sequences
#    We use one_hot()

# 4.1 one_hot() is a swiss-knife. Full syntax is:
#     one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
#     Try: one_hot("This is a this sentence", 10)    # 10 is vocab_size
#     one_hot() is NOT ONE HOT encoding but encoding into an integer between [1,vocab_size]
encoded_docs = []
for doc in texts:
    # 4.1.1 For each comment, convert to sequence
    s = one_hot(doc,vocab_size)
    # 4.1.2 And append
    encoded_docs.append(s)

# 4.1.3 Check two of these sequences
encoded_docs[:2]


# 5.0 Let us limit our sentence length
#     So that all our documents can be of same input_length
#     or vector-size

data = pad_sequences(encoded_docs,
                            maxlen=max_sentence_length,
                            padding='post')

# 5.0.1
data
data.shape           # 2000 X 200, as expected


################# EE. Split data into train and test

# 5.1 Shuffle sequence so as to take random sample
data.shape              # (2000,200)
indices = np.arange(data.shape[0])

#5.2 Shuffle this sequence
np.random.shuffle(indices)
indices

# 5.3 Extract data and corresponding labels
data = data[indices, ]
labels = labels[indices]

# 5.4 Split data into train and val
X_train = data[:training_samples, ]
y_train = labels[:training_samples]     # Labels

x_val = data[training_samples:training_samples+validation_samples,]
y_val = labels[training_samples:training_samples+validation_samples]

# 5.5 If output layer is softmax and we have two neurons at the output,
#     our output labels must hold two values
#     On the other hand if output layer is sigmoid and just one
#     neuron, the following conversion is NOT NEEDED
y_train =to_categorical(y_train, num_classes=2)
y_val =to_categorical(y_val , num_classes=2)


################# FF. Build model now

# 6 Build a simple sequential model
model = Sequential()
model.add(Embedding(input_dim = vocab_size,
                    output_dim = 80,              # Each word's vector size
                    input_length = max_sentence_length
                    )
          )

# 6.1
#    No of output neurons: 100. Input is fixed.
model.add(LSTM(100))
model.summary()

# 6.2 Note the model summary. Its size is (max_sentence_length X vector length)
#     That is like an image, one sentence is : [35, 15, 12, 7]
#     Vector formulation is (4 X 8):
#                     1.2  2.3  3.6  4.9  7    9.1  4.4 8.8
#                     0.9  4.4  1.6  2.2  4.1  1.3  4.5 2.9
#                     9.1  2.2  1.9  1.2  3.4  1.3  6.0 7.7
#                     0.08 9.1  3.6  7.2  5.8  1.6  4.5  0.2
#
#     After flattening it, it will be of size 32

# 6.3 Output layer
model.add(Dense(2,activation = 'softmax'))

# 6.4 compile
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics =['acc'])


# 6.5 Learn classification as also word_vector with back propogation
history = model.fit(X_train,
          y_train,
          epochs = 50,
          verbose = 1
          )

# 6.6
loss, accuracy = model.evaluate(x_val, y_val, verbose = 0)
loss
accuracy               # 73.4%

# 6.7 Here are the learnt word-vectors.
#     Each vector is of length 8
model.layers[0].get_weights()            # List of arrays
len(model.layers[0].get_weights())
model.layers[0].get_weights()[0].shape   # (50,8)



########################################################
