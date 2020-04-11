"""
Last amended: 13/12/2018
My folder: /home/ashok/Documents/10.nlp_workshop

Ref:
https://github.com/hundredblocks/concrete_NLP_tutorial
https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb

DataSource: Disaster events on social media
  	Contributors looked at over 10,000 tweets retrieved with
  	a variety of searches like “ablaze”, “quarantine”, and
  	“pandemonium”, then noted whether the tweet referred to
  	a disaster event (as opposed to used in a joke or with
  	the word or a movie review or something non-disastrous).

Objective:
          Try to correctly predict tweets that are
          about disasters using word2vec

    (gensim is not installed on theano environment)

	$ source activate tensorflow
	$ ipython

"""

#################### AA. Call libraries ####################
## 1.0 Clear memory and call libraries
%reset -f

# 1.1 Array and data manipulation libraries
import pandas as pd
import numpy as np

# 1.2 nltk is a leading platform for building Python programs to work
#      with human language data. Tokenize sentence into words as per
#      given regular expressions
from nltk.tokenize import RegexpTokenizer
#  For kera Tokenizer, pl see your file:
#    /home/ashok/Documents/8.rnn/keras_embedding_tokenizer.py
#from keras.preprocessing.text import Tokenizer

# 1.3 Data modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1.4 Accuracy reports/metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 1.5 For word2vec manipulations
#     (gensim is not installed on theano environment)
import gensim

# 1.6 Utilities
import os,time

#################### BB. Sanitize Inputs ####################

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

# 2.4.1 choose_one and class_label carry the same information
questions[['choose_one','class_label']].loc[15:35, :]



# 2.5 Clean the data
#     Let's use a few regular expressions to clean up data,
#     and save it back to disk for future use
# Ref: http://www.regexlib.com/(X(1)A(5VYROh19ihtNP-JfoQjDSiymxutkQHoWK2wjkPydrzYhq5c450yDq5RxRfvQsrh09BHiCyJCNwIztpf_CsfLdb_1KDUgs1fy1BApFZ3xE8T_lQnsDlHhP1OEtO4T58M2x4_sJkRwe2gdtYRimfwin6FDQLaahvTmhUwWQ8sBhDg09H7kKKUevqzC8N1j4LNT0))/cheatsheet.htm?AspxAutoDetectCookieSupport=1
#     Experiment like this:
#
#          "@abc".replace(r"@", "at")
#
def standardize_text(df, text_field):
    # 2.5.1 Matches http followed by one or more non-white-space characters
    #       ie. including all the alphabetical and numeric characters, punctuation, etc.
    #       str() is pandas method. r"http\S+" implies regular expression
    #       \S is any non-whitespace character
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
   	# 2.5.2 Matches http. Note the sequence of instructions
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

#################### CC. Tokenize text ####################

# 3.0 Our data is clean, now it needs to be prepared
#     Tokenizing sentences to a list of separate words
#     Creating a train test split
#     Inspecting our data a little more to validate results


# 3.1 `RegexpTokenizer`: Splits a string into substrings as per a regular expression.
#                        and returns a list of tokens
#      Instantiate RegexpTokenizer object with a regular expression
tokenizer = RegexpTokenizer(r'\w+')

# Examples:
#
#	capword_tokenizer = RegexpTokenizer('[A-Z]\w+')
#	capword_tokenizer.tokenize("Good people are Rare")
#
#	tokenizer = RegexpTokenizer('\w+')     # \w: Matches any word character.
# 	tokenizer.tokenize("Good muffins cost a lot\nin New York.  Please buy me\njust two of them.\n\nThanks.")

# 3.2 'apply()'' the tokenizer to each entry on column 'text' of clean_questions
#     Add a column 'tokens'
clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
clean_questions.head()


# 3.3 Get each row of 'text' into a list of lists. So length of list
#     will be same as that of number of rows.
#     Also get a list of class_labels

# 3.3.1 First 'text' columns
#       tolist(): Pandas Series method that returns a list of the values
list_corpus = clean_questions["text"].tolist()

len(list_corpus)               # 10876
clean_questions.shape          # (10876,5)

list_corpus[0]

# 3.3.2 Compare with 'tokens' column
clean_questions['tokens'][0]


# 3.3.3
list_labels = clean_questions["class_label"].tolist()
list_labels[0]
len(list_labels)

#################### DD. Read word2vec and experiment ####################

## 4.
# Every 'word' is projected into a multi-dimensional space. For each word we will take
# Load 300-dimensioned slimmed-down version of actual model
#  Actual model's file size is many GBs (even though dimensions are 300)
#   Actual model has words from other languages such as Chinese
# 4.1 Where is my word2vec file
word2vec_path = "/home/ashok/.keras/datasets/google_word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz"
# 4.2 Load model into memory
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# 5. So what is the vector for 'dog'
word2vec['dog']
len(word2vec['dog'])
# 5.1 And for 'cat'
word2vec['cat']
# 5.2 And for 'a'
word2vec['a']
# 5.3 Check if a word is in word2vec
'dog' in word2vec        # True
'a'   in word2vec        # False
',' in word2vec          # False

# 5.4 Just some fun with word2vec
word2vec.most_similar(positive=['woman', 'king'], negative=['man'])
word2vec.most_similar(positive=['woman', 'uncle'], negative=['man'])
word2vec.doesnt_match("paris berlin london austria".split())
word2vec.doesnt_match("man woman child kitchen".split())
word2vec.doesnt_match("man woman child cow".split())
word2vec.doesnt_match("france england germany berlin".split())

## 6. So, What do we want to do for each comment:
#     For every word in a comment, find its coordinates in a 300-dimensioned space
#     For each dimension, sum up all dimension-values. Then take an average.
#     Example:     Let comment be  : ['dog','and','cat']
#     Let each word's coordinate in 3-dimension be:
#                           'dog' :   [3, 5, 10]      # Usually 300 dimension
#                           'and' :   [1, 4,  2]
#                           'cat' :   [5, 9, 15]
#        Avg direction of sentence:   [9/3, 18/3,  27/3] = [3,6,9]
# 6.1 Here are steps:
## 6.1.1 Case 1
k= 300
# 6.2 Let our list be:
mylist1 = ['dog','cat','a']
# 6.3 If our word is in word2vec, get its dimensions from word2ve,
#     else create a list of 300-zeros
# vectorized = [word2vec[word] if word in word2vec else np.zeros(k) for word in mylist1 ]
vectorized = []
for word in mylist1:
    if word in word2vec:
        vectorized.append(word2vec[word])
    else:
        vectorized.append(np.zeros(k))

# 6.4
vectorized
# 6.5
length = len(vectorized)
length     # 3

# 6.6
# axis = 0 is running vertically downwards
# axis = 1 is running  horizontally across columns
summed = np.sum(vectorized,axis = 0)
# 6.7
averaged1 = np.divide(summed, length)
averaged1

## 6.1.2 Case 2
mylist2 = ['dog','a']
vectorized = [word2vec[word] if word in word2vec else np.zeros(k) for word in mylist2 ]
vectorized
length = len(vectorized)
summed = np.sum(vectorized,axis = 0)
averaged2 = np.divide(summed, length)
averaged2


#################### EE. Define useful functions and expt ####################

# 7.  We put all above steps in a function
# 7.1 Given a list of tokens, function to return avg direction
def get_average_word2vec(tokens_list, word_2_vec, k=300):
	"""
    tokens_list:  One list of words/tokens. Example: ['dog','cat']
    vector: Usually: word2vec
    k: Number of dimensions of word2vec space. For Google it is 300
	"""
	# 7.1.1 An empty list of word vectors
	vectorized = []
	# 7.1.2 Loop through each word in the list
	for word in tokens_list:
		# 8.1.3 If the word is in the word2vec
		if word in word_2_vec:
			vectorized.append(word_2_vec[word]) # Append list of word-vectors. List len = 300
		else:
			vectorized.append(np.zeros(k))     # Append list of zero-vectors
											   # vectorized: ['dog','cat'] :[[3, 5, 10],[5, 9, 15] ]
											   # vectorized: ['dog','a']   :[[3, 5, 10],[0, 0, 0 ] ]
	# 7.1.4 Get length, sum and average
	length=len(vectorized)                     # Two in above case of [[3, 5, 10],[0, 0, 0 ]]
	summed = np.sum(vectorized,axis = 0)       # summed: [3,5,10]   for ['dog','a']
	averaged = np.divide(summed,length)        # averaged: [3/2,5/2,10/2]
	return averaged

## 8. USing above function

# 8.1 Apply the function to the two lists
get_average_word2vec(mylist1, word2vec,  300)
get_average_word2vec(mylist2, word2vec,  300)


## 9. Getting average vectors from a column of dataframe
# 9.1 Let us create just one cell dataframe having a list in that cell
ex = {'col1' : [['dog','cat']]}
xx= pd.DataFrame(ex)
xx
vectors= word2vec

# 9.2 Apply get_average_word2vec() over every cell in the column
xx['col1'].apply(lambda x: get_average_word2vec(x, vectors))

# 9.3 Comapre above result with earlier result
averaged1


# 9.4 Next, we create two-cell dataframe having a list in each cell
ex1 = {'col1' : [['dog','cat'], ['cow','goat']], }
xx1= pd.DataFrame(ex1)
xx1
vectors= word2vec

# 9.5 Outputs
embeddings = xx1['col1'].apply(lambda x: get_average_word2vec(x, vectors))
embeddings           # Pandas series. Each of the two sells is ndarray of size 300


# 9.6
list(embeddings)     # A list of two lists


## 10. Define a function to encapsulate above function & code
#     to get cell-wise average in a column
def get_word2vec_embeddings(vectors, clean_questions):
  """
  vectors: Usually word2vec
  clean_questions: Cleaned dataframe
  Returned object is a list containing as many lists as there are rows in the dataframe
  """
  embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors))
  return list(embeddings)


# 10.1 Invoke the function now
embeddings = get_word2vec_embeddings(word2vec, clean_questions)
len(embeddings)           # Same as number of rows
embeddings[0]
type(embeddings[0])       # numpy.ndarray
len(embeddings[0])        # 300

# So embeddings is a list of arrays. Each array
# is an averaged array of 300 elements

#################### FF. Modeling ####################
#### Modeling ####

X = embeddings
y = list_labels

# 10. Split embeddings into two parts.
# 'train_test_split' can also take 'list' as input besides dataframe
#    Allowed inputs to it are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(X,y,
	                                                                                    test_size=0.2,
	                                                                                    random_state=40
	                                                                                    )
# 10.1 Instantiate object to perform logistic regression.
#      Note here there are more than 2-classes
clf_w2v = LogisticRegression(C=30.0,                   # Inverse of regularization strength;  smaller values specify stronger regularization.
                             penalty = 'l2',           # Default: ‘l2’
                             class_weight='balanced',  # “balanced”: Adjust weights inversely proportional to class frequencies as n_samples / (n_classes * np.bincount(y))
                                                       #       n_samples = 100   ; classes ['s1', 's2']
													 #       s1 = 25
                                                       #       s2 = 75
                                                       #       s1 weight: 100/(2 * 25)  : 2
                                                       #       s2 weight: 100/(2 *75)   : 2/3
                             solver='newton-cg',  # Algorithm for optimization problem; for multiclass problems select: ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
                             multi_class='multinomial',
                             random_state=40,
                             max_iter = 1000,           # default: 100
                             n_jobs = -1,               # Use all cores for parallel operation
                             verbose = 2
                             )

# 10.2 Fit the object to data. Develop model
clf_w2v.fit(X_train_word2vec, y_train_word2vec)

#################### GG. Prediction ####################
#### Prediction and Evaluation ####

# 11 Make predictions of 'test' data
y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)

# 11.1 Just check number of unique classes in predicted output
np.unique(y_predicted_word2vec)


# 12. Define a function to get all accuracy metrics
def get_metrics(y_test, y_predicted):
    # 12.1 'weighted': Calculate metrics for each label, and find their average,
    #       weighted by support (the number of true instances for each label).
    precision = precision_score(y_test, y_predicted,
                                pos_label=None,          # No preference for any sentiment
                                average='weighted'
                                )

    recall = recall_score(y_test, y_predicted,
                          pos_label=None,
                          average='weighted'
                          )

    f1 = f1_score(y_test, y_predicted,
                  pos_label=None,
                  average='weighted'
                  )

    accuracy = accuracy_score(y_test, y_predicted)

    # 12.3 Return all four scores
    return accuracy, precision, recall, f1


# 13. Get and print accuracy results
#     Ref for print(): https://pyformat.info/
accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec, y_predicted_word2vec)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec))

# 13.2 And confuson matrix
# Count of true negatives is `C_{0,0}`, false negatives is 'C_{1,0}`
# true positives is `C_{1,1}` and false positives is :math:`C_{0,1}`
confusion_matrix(y_test_word2vec, y_predicted_word2vec)

######################################################################
