"""
Last amended: 28th June, 2019
"""

# b. Embedding and Tokenizer in keras *****
#    http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
# Objectives:
#   Keras has some classes targetting NLP and preprocessing text
#   but it’s not directly clear from the documentation and samples
#   what they do and how they work. So here are some simple examples
#   to expose what is going on.
#   Besides, gensim and nltk can also be used for tokenization.
#   See file: 8.rnn/0.document_to_id_conversion.py


### Expt 1
##################################################################################
##************Tokenize text to integer sequences***************************
##################################################################################

%reset -f
import os
import numpy as np

# 1. The Tokenizer class in Keras has various methods
#     which help to preprocess text so that after preprocessing
#      text can be used in neural network models.
#    https://keras.io/preprocessing/text/
from keras.preprocessing.text import Tokenizer

# 2. Change folder path and read file
#    os.chdir("C:\\Users\\ashokharnal\\OneDrive\\Documents\\recurrentNeuralNetwork")

# 3. Instantiate Tokenizer class. Create Tokenizer
#      object which will take into account nb_words
#       features, those with max freq.
#    Tokenizer class allows to vectorize a text corpus,
#    by turning each text into either a sequence of integers
#    (each integer being the index of a token in a dictionary)
#    or into a vector where the coefficient for each token could
#    be binary, based on word count, based on tf-idf...

nb_words = 4    # Our corpus will contain just (4-1 ie 3)
                #    top-words with max freq. Just demo.
                #      See encoded_docs below

# num_words: the maximum number of words to keep,
#  based on word frequency.
#  Retain just the most common num_words words.
# 3.1 Class instatniation

# 3.1.1 Each one of these chars shall be removed
#       from documents while tokenizing

chars_tobe_filtered = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ '  # All punctuations and non-alphabets
chars_tobe_filtered = chars_tobe_filtered + '0123456789'     # All digits be also removed

tokenizer = Tokenizer(num_words=nb_words,  # max num of words to keep, based on word freq
                      filters=chars_tobe_filtered,  # a string where each element
                                                                   #  is a character that will be
                                                                   #   filtered out from the texts.
                      lower=True,          # Convert to lower case
                      split=' ',           # Where to split each word
                      char_level=False,    # if True, every character will be treated as a token.
                      oov_token=None       # Out of vocabulary (oov) text
                                           #  What happens when a sentence is later presented
                                           #    that does contain any word from orginal vocab
                                           #    as for example, 'morning'
                      )


# 4. Prepare an internal dictionary of words: fit_on_texts method
#    is the training phase. Punctutations are removed. tolower is default.
#     Also 'nb_words' does not limit dictionary size to 4.
#     Word-frequencies of some of the words are:
#     shines: 5, water: 4,  star: 3, is: 2, sun: 1,
#     flowing: 2, in:1, world: 1

texts = ["1. Sun shines brightly  in June!",
         "2. Star light shines on water?",
         "Water was flowing.",
         "Flowing water, shines",
         "Galaxy or star?",
         "Shoe shines ",
         "Star also shines",
         "water is life",
         "World is energy"]

"""
fit_on_texts:
    It Updates internal vocabulary based on a list of texts.
    This method creates the vocabulary index based on word
    frequency. So if you give it something like, "The cat
    sat on the mat." It will create a dictionary s.t.
    word_index["the"] = 0; word_index["cat"] = 1 it is
    word -> index dictionary so every word gets a unique
    integer value. Lower integer value means more frequent
    word (often the first few are punctuation because they
    appear a lot).
"""

# 4.1 Learn 'texts'
tokenizer.fit_on_texts(texts)

"""
Study tokenizer attributes:
  word_counts: A dictionary of words and their counts.
  word_docs:   A dictionary of words and how many documents each appeared in.
  word_index:  A dictionary of words and their uniquely assigned integers.
  document_count:An integer count of the total number of documents that were used to fit the Tokenizer.
"""

# 4.2 Words have indexing associated. Index number is as per frequency.
#     Less the index value more the frequency ('shines' has max freq)

dic = tokenizer.word_index
dic

# 4.3 Sort dictionary by its value
#     Words higher up have more frequency
sorted(dic, key =dic.get )


# 4.3. tokenizer is also aware of no of sentences
#      (or rather no of elements in the list)
tokenizer.document_count

# 4.4. Get a dictionary of words and their counts
tokenizer.word_counts

# 4.5. Or whether lower-casing was applied and how many sentences
#      were used to train:
tokenizer.lower

# 5. Tokenized text with only 3 (ie nb_words-1) words or integers

tokenizer.texts_to_sequences(texts)

# 5.1 How the nine documents have been encoded
#     mode: one of "binary", "count", "tfidf", "freq".
encoded_docs =tokenizer.texts_to_matrix(
	                                    texts =texts,
	                                    mode='count'
	                                    )

# 5.2 Sum of 1s in each column gives that word frequency
#     The array is for 'num_words' columns
#     Four words and there frequencies are:
#           'shines' (5), 'water' (4), 'is' (4), 'sun' (3)
#     Try also with nb_words = 5

encoded_docs    # Interpre the output as follows

"""
Interpret encoded_docs as below:

               shines  water star
    array([[0.,   1.,   0.,   0.],
           [0.,   1.,   1.,   1.],
           [0.,   0.,   1.,   0.],
           [0.,   1.,   1.,   0.],
           [0.,   0.,   0.,   1.],
           [0.,   1.,   0.,   0.],
           [0.,   1.,   0.,   1.],
           [0.,   0.,   1.,   0.],
           [0.,   0.,   0.,   0.]])
    Total         5     4     2

"""
# 5.3  Get tfidf values
encoded_docs =tokenizer.texts_to_matrix(
	                                    texts =texts,
	                                    mode='tfidf'
	                                    )

# 5.4  Again for the top-three words
encoded_docs


# 6. Convert text to sequences.
#    Only indicies less than or equal to (nb_words -1) (4-1) kept.
#    Try also with nb_words = 5
tokenizer.word_index
tokenizer.texts_to_sequences(["1Sun shines in morning water "])    # 'morning' is out of vocabulary (oov)
                                                                  # [[1, 2]]    'shines' 'water'
tokenizer.texts_to_sequences(["Sun shines is morning water "])    # [[1, 3, 2]]  'shines' 'water', 'is'
tokenizer.texts_to_sequences(["shines in morning water "])        # [[1, 2]]
tokenizer.texts_to_sequences(["in morning water "])               # [[2]]


# 7. A parameter-less constructor yields full sequences:
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
tokenizer.word_index


# 7.1 Based on dictionary convert following sentence to
#      integer sequence. As 'during' and  'day' words are absent
#      in dictionary (OOV), these are not converted to sequences
tokenizer.texts_to_sequences(["Sun shines in flowing water during day"])
tokenizer.word_index


# 8. To use sentences for analysis one can’t use arrays
#       of variable lengths, corresponding to variable
#        length sentences. So, the trick is to use the
#         texts_to_matrix method to convert the sentences
#          directly to equal size arrays:
# 8.1
np.set_printoptions(threshold=np.nan)   # Do not truncate screen displays
tokenizer.texts_to_matrix(texts)        # Read 11.2 below to understand
tokenizer.texts_to_matrix(texts).shape  # 6 X 13; 6 sentences and 12 words


# 8.2  To discover which word is the matrix column header,
#       prcoeed as follows, Note that column-wise sum decreases
#       from left-to-right
"""
The first column of matrix is all zeros
Next check second col of matrix (col-sum= 4):
    Which word is common in sentences: 1, 2, 4, 6? Ans: shines
Check third col (col-sum = 3):
    Which word is common in sentences 2,3,4? Ans: water
Check last col:
#     Which word is unique to last sentence? Ans: World
And so on...
"""

### Expt 2
##################################################################################
##************Padded integer sequence to common length***************************
##################################################################################

# Ref:
# http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/
"""
We’ll have a look at how Keras’ tokenization and sequence padding works
on some toy data, in order to work out what’s going on under the hood.


Keras represents each word as a number, with the most common word in
a given dataset being represented as 1, the second most common as a 2,
and so on. This is useful because we often want to ignore rare words,
as usually, the neural network cannot learn much from these, and they
only add to the processing time. If we have our data tokenized with the
more common words having lower numbers, we can easily train on only the
N most common words in our dataset, and adjust N as necessary (for larger
datasets, we would want a larger N, as even comparatively rare words will
appear often enough to be useful).

Tokenization in Keras is a two step process. First, we need to calculate
the word frequencies for our dataset (to find the most common words and
assign them low numbers). Then we can transform our text into numerical
tokens. The calculation of the word frequencies is referred to as
‘fitting’ the tokenizer, and Keras calls the numerical representations
of our texts ‘sequences’.
"""

%reset -f
import os
import numpy as np

# 1. The Tokenizer class in Keras has various methods
#     which help to prepare text so it can be used
#      in neural network models.
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 2. Change folder path and read file
# os.chdir("C:\\Users\\ashokharnal\\OneDrive\\Documents\\recurrentNeuralNetwork")

# 3. In line one, we create a tokenizer and say that it should ignore all
#    except the five most-common words (in practice, we’ll use a much
#    higher number).
#    In line three, we tell the tokenizer to calculate the frequency of
#    each word in our toy dataset.
#    In line four, we convert all of our texts to lists of integers

tokenizer = Tokenizer(num_words=5)
toytexts = ["Is is a common word", "So is the", "the is common", "discombobulation is not common"]
tokenizer.fit_on_texts(toytexts)
sequences = tokenizer.texts_to_sequences(toytexts)
print(sequences)

# We can see that each text is represented by a list of integers.
# The first text is 1, 1, 4, 2. By looking at the other sequences,
# we can infer that 1 represents the word “is”, 4 represents “a”,
# and 2 represents “common”. We can take a look at the tokenizer
# word_index, which stores to the word-to-token mapping to
# confirm this:

print(tokenizer.word_index)

# Rare words, such as “discombobulation” did not make the cut of
#  “5 most common words”, and are therefore omitted from the
#    sequences. You can see the last text is represented only
#     by [1,2] even though it originally contained four words,
#      because two of the words are not part of the top 5 words.

# Finally, we’ll want to “pad” our sequences. Our neural network
#  can train more efficiently if all of the training examples are
#   the same size, so we want each of our texts to contain the same
#    number of words. Keras has the pad_sequences function to do
#     this, which will pad with leading zeros to make all the texts
#      the same length as the longest one:
padded_sequences = pad_sequences(sequences)
print(padded_sequences)

# The last text has now been transformed from [1, 2] to [0, 0, 1, 2]
#  in order to make it as long as the longest text (the first one).
