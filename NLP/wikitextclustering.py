# -*- coding: utf-8 -*-
"""
Last amended: 28th June, 2019
My folder: C:\Users\ashok\OneDrive\Documents\sentiment_analysis
           /home/ashok/Documents/10.nlp_workshop/text_clustering

# Regular expressions, how to:
    http://203.122.28.230/moodle/mod/url/view.php?id=1842
    https://docs.python.org/3/howto/regex.html#regex-howto


Virtual Machine: lubuntu_deeplearning_1

Objectives:
    i)   How to assemble text from files
         or pandas dataframe for cleaning
    ii)  How to clean text
    iii) How to stem text
    iv)  How to transform text to Tfidf format
    v)   Text clustering of wiki documents


"""

###################### 1. Call libraries #####################
# 1.0 Clear memory
%reset -f
# 1.1 Array and data-manipulation libraries
import numpy as np
import pandas as pd

# 1.2 sklearn modeling libraries
# 1.2.1 For calculating tf-idf values
#       https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1.3 For stemming words
from nltk.stem.porter import PorterStemmer


# 1.4 Text processing
# 1.4.1 Import 're' module for
#       regular expression matching
import re
import os

# ===== Begin

# 1.5 Where are my text files to be clustered
os.chdir("/home/ashok/.keras/datasets/textclustering")
os.listdir()   #  _p: philosophy document,
               #  _l: Law document,
               #  _r: Religious document,
               #  _q: Quantum mechanics

# 2.0 Cleaning text using regular expression
#     Examples of usage of re module for text cleaning
"""
https://docs.python.org/3/library/re.html#re.sub
Syntax:
 re.sub(pattern, repl, string, count=0, flags=0)

    pattern: Regular expression for pattern matching
    string:  Your string that you want to clean
    repl:    Replace unclean characters in string with 'repl'

    Returns the 'string' obtained by replacing the
    leftmost non-overlapping occurrences of 'pattern'
    in string by the replacement 'repl'. If the
    'pattern' isn’t found, 'string' is returned unchanged.
    'repl' can be a string or a function; if it is a
    'string', any backslash escapes in it are processed.
    That is, \n is converted to a single newline character,
    \r is converted to a carriage return, and so forth.
"""


# 2.1 Replace bracketed numbers with space

x = "[8]OK good[6] [6] [5]done"
result= re.sub(r'[\[0-9\]]',' ', x)
result

# 2.2 Replace newline (\n) with space

x = "OK \n good\n  \ndone"
result= re.sub('\n',' ', x)
result

# 2.3 Replace " 's " with space

x="After that it's just a matter "
result= re.sub('\'s',' ', x)
result

# 2.4 Remove all html or other tags
# https://stackoverflow.com/a/3075532
x = " <title>Cultural universal</title>      <ns> </ns>      <id>       </id>      <revision>        <id>         </id>        <parentid>         </parentid>        <timestamp>    -  -  T  :  :  Z</timestamp>"
clean = re.compile('<.*?>')    # re.compile() creates a reusable object 'clean'
                               #  that can be used in place of pattern
re.sub(clean, '', x)

# 3.0 Example of processing a single text file
# 3.1 Open text file
text_file = open('l1_q.txt', "r",  encoding="utf8")
# 3.2 Read it line by line. All lines become string elements
#     of a list
tx = text_file.readlines()
tx
# 3.3
len(tx)          # 3
type(tx)         # list
type(tx[0])      # str

# 3.2.1 Join all lines in the list and create a single string
tx = " ".join(tx)
tx
type(tx)     # str

# 4. Next read all files, one by one, and clean them up
#    Each file, after cleaning, is one string. All strings
#    are then appended to a list 'lines'. So 'lines' contains
#    as many strings, as there are files.

file_List = os.listdir()
file_List

# 4.1 Open one file from file_List
#     Read each line of file as a string.
#     Join all strings (lines) into one string
#     Append the combined string to a list
#     Now read the next file and repeat the process
#     List now contains each file as one element
#      and as many elements as there are files
def readFiles(fileList):
    lines = []
    # 4.1.1 For every file in the directory:
    for i in fileList:
        # 4.1.2 Open it and coalesce into one string
        text_file = open(i, "r",  encoding="utf8")
        tx = text_file.readlines()
        tx = " ".join(tx)
        lines.append(tx)
    return(lines)


# 4.2 Read all files now and create a list of strings
out = readFiles(file_List)
out
out[0]       # First file
len(out)     # 12

# 4.3 Clean the list of strings
def cleanTxt(listOfStrings):
    lines = []
    for tx in listOfStrings:
        # 4.3.1 Clean each string through a series
        #     of cleaning operations
        clean = re.compile('<.*?>')
        tx = re.sub(clean, '', tx)
        # 4.3.2 Replace bracketed numbers with space
        tx= re.sub(r'[\[0-9\]]',' ', tx)
        tx= re.sub('\n',' ', tx)
        tx= re.sub('\'s',' ', tx)
        tx= re.sub('\'s',' ', tx)
        # 4.3.3 Replace URLs
        tx = re.sub(r'^https?:\/\/.*[\r\n]*', '', tx, flags=re.MULTILINE)
        tx = re.sub('[*|\(\)\{\}]', " ",tx)
        tx = re.sub('[=]*', "",tx)
        # 4.3.4 Replace other tags generally part of a web-file
        clean = re.compile('&lt;')
        tx = re.sub(clean, '', tx)
        clean = re.compile('&gt;')
        tx = re.sub(clean, '', tx)
        clean = re.compile('&quot;')
        tx = re.sub(clean, '', tx)
        lines.append(tx)
    return lines

lines = cleanTxt(out)
lines


# 5. Check what have we got
type(lines)       # List
lines             # Text elements in list
len(lines)        # How many lines? or files?  12


# 6. Stemming text
#    Instantiate PorterStemmer object
porter_stemmer = PorterStemmer()

# 6.1 Define a function to use NLTK's PorterStemmer
def stemming_tokenizer(str_input):
    words = str_input.split()
    words = [porter_stemmer.stem(word) for word in words]
    return words


# 7.0 Let us stem our list of string elements
new_lines = []
for line in lines:
    stemmed_line = stemming_tokenizer(line)
    stemmed_line = " ".join(stemmed_line)    # Join words back into a string
    new_lines.append(stemmed_line)


# 7.1 Observe results.
new_lines
len(new_lines)
new_lines[0]


## 8.0 Transform text to tf-idf matrix
#  8.1 Instantiate TfidfVectorizer object
#      Instead of taking stemmed text, we will
#      take another approach below:

vec = TfidfVectorizer(use_idf=True,
                      strip_accents = 'unicode', # Remove accents during preprocessing step.
                      lowercase = True,
                      tokenizer=stemming_tokenizer,
                      max_features = 1000,   # Consider only top frequent features
                      stop_words='english'  # Remove stop-words
                      )


# 8.2 Use 'vec' object to transform:
data = vec.fit_transform(lines)
data      # 12x500 sparse matrix of type '<class 'numpy.float64
data.shape


# 8.3 Put it sparse data in pandas data frame with feature_names
#     Just to observe what we got. No other purpose.
df = pd.DataFrame(data.toarray(), columns=vec.get_feature_names())
df.head()


# 9. Apply kmeans now
number_of_clusters=  4
# 9.1 KMeans object
km = KMeans(n_clusters=number_of_clusters,
            max_iter=500
            )

# 9.2 Fit the array directly
km.fit(data)
# 9.3 Have a look at labels
km.labels_


# 9.4 Put filenames and corresponding clusters in a dataframe
results = pd.DataFrame()            # An empty dataframe
results['filename'] = os.listdir()  # One column gets filenames
results['cluster'] = km.labels_     # Another corresponding cluster labels
results.sort_values('cluster')      # Sort by cluster labels and display

################# DataFrame to a List ######################

# 10. Given a dataframe of text, how to get a
#     list of string elements
import pandas as pd
dfx = pd.DataFrame({ "text" : [  " This is bad", "This is good", "This is great", "This is OK"]  })
dfx

# 10.1 Define a simple function to do the job
def df_to_list(df, col_name):
    fin= df[col_name].tolist()
    return fin

# 10.2 Now get a list of elements from dataframe
df_to_list(dfx,'text')

################### I am done #################################
