# Last amended: 15/12/2018
# Myfolder:
# Ref: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

# Objective: Understanding how Word Embedding Layer of Keras works
#            Train embedding layer to transform text to vectors

"""
Steps:
       a. Prepare a list of +ve and -ve sentences
       b. Clean text
       c. Prepare a list of corresponding labels (ex: [0,1,0,0]
       d. Transform all documents to sequences (use one_hot())
       e. Pad sequences to fixed length
       f. Build Embedding layer + Classification model
       g. Train and find accuracy on train-set itself
"""


"""
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

# 1.0 Call libraries
%reset -f
import numpy as np

# 1.1 Keras text and sequence processing
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

# 1.2 Keras modeling layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

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

# 2.1 How many unique words exist?
st = ""
for doc in docs:
	doc = doc.replace('!', "")
	doc = doc.replace('.', "")
	st = st + " " + doc

# 2.2
len(st.split(' '))            # Total words: 21
len(set(st.split(' ')))       # Unique 17


# 3. Define array of sentiment class labels for documents
labels = np.array([0,0,0,0,0,1,1,1,1,1])

# 3.1 Let our vocabulary size be maximum 50
vocab_size = 50

# 4. Transform all documents to integer sequences
#    We use one_hot()

# 4.1 one_hot() is a swiss-knife. Full syntax is:
#     one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
#     Try: one_hot("This is a this sentence", 10)    # 10 is vocab_size
#     one_hot() is NOT ONE HOT encoding but encoding into an integer between [1,vocab_size]

encoded_docs = [one_hot(doc,vocab_size) for doc in docs ]
encoded_docs

# 4.1.1 OR do it, as:
encoded_docs = []
for doc in docs:
	# 4.1.2
	out = one_hot(doc,vocab_size)
	# 4.1.3
	encoded_docs.append(out)


# 5.0 Let us limit our sentence length
#     So that all our documents can be of same input_length
#     or vector-size
max_sentence_length = 4
X_train = pad_sequences(encoded_docs,
                            maxlen=max_sentence_length,
                            padding='post')
# 5.1
X_train

# 5.2 Build a simple sequential model
model = Sequential()
model.add(Embedding(input_dim = vocab_size,
                    output_dim = 8,              # Each word's vector size
                    input_length = max_sentence_length
                    )
          )

# 5.3
#
model.add(Flatten())

# 5.4 Note the model summary. Its size is (max_sentence_length X vector length)
#     That is like an image, one sentence is : [35, 15, 12, 7]
#     Vector formulation is (4 X 8):
#                     1.2  2.3  3.6  4.9  7    9.1  4.4 8.8
#                     0.9  4.4  1.6  2.2  4.1  1.3  4.5 2.9
#                     9.1  2.2  1.9  1.2  3.4  1.3  6.0 7.7
#                     0.08 9.1  3.6  7.2  5.8  1.6  4.5  0.2
#
#     After flattening it, it will be of size 32

model.summary()
model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics =['acc'])

# 5.4 It is important to note the size of flattened layer. Each word
#      is a vector of size 8. In a sentence thare are four words so
#       for a sentence, size of flattened layer is 4 X 8 = 32. Each
#        vector is in turn learnt weights of Embedding layer (layer 0)
#         for that word.

model.summary()

# 5.5 Learn classification as also word_vector with back propogation
model.fit(X_train,
          labels,
          epochs = 50,
          verbose = 1
          )

# 5.6
loss, accuracy = model.evaluate(X_train, labels, verbose = 0)
loss
accuracy

# 5.7 Here are the learnt word-vectors.
#     Each vector is of length 8
model.layers[0].get_weights()            # List of arrays
len(model.layers[0].get_weights())
model.layers[0].get_weights()[0].shape   # (50,8)



########################################################
