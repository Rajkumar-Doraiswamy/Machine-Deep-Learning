# Last amended: 28th June, 2019
# Ref: https://radimrehurek.com/gensim/tut1.html#from-strings-to-vectors
#
# Objective:

#         A. Convert tokens with each document to corresponding
#            'token-ids' or integer-tokens.
#            For text cleaning, pl refer wikiclustering file
#            in folder: 10.nlp_workshop/text_clustering
#            This file uses gensim for tokenization
#         B. Keras also has  Tokenizer class that can also be
#            used for integer-tokenization. See file:
#            8.rnn/3.keras_tokenizer_class.py
#         C. nltk can also tokenize. See file:
#            10.nlp_workshop/word2vec/nlp_workshop_word2vec.py 


%reset -f

# 1.1  gensim contains tools for Natural Language Processing
#      Module 'corpora' contains sub-modules and methods to
#      work with text documents
from gensim import corpora

# 1.2 defaultdict is like an ordinary dict. Only that if a key does
#  not exist in the dict, then on its search it inserts that 'key'
#   with a value that is defined by an initialization function (such as int())
from collections import defaultdict

# 2. Create a sample collection (list) of documents
#    See text_clustering.py file as to how to get this list
#    from folder of files or pandas dataframe
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# 2.1 Clean documents: See file text_clustering.py

# 2.2 Stem documents : See file text_clustering.py

# 2.3 Create an arbitrary list of stopwords that we do not want
#     Detailed list of english stopwords is available at:
#     https://gist.github.com/sebleier/554280
stoplist = set('for a of the and to in'.split())


# 3. Tokenize ie parse into words
#      each document in the document-collection
def tokenize(docs):
    tokenized = []          # This will be a list of lists
    for document in docs:   # For each senetence in the document-collection
        tokenized_document = []
        for word in document.lower().split():
            if word not in stoplist:
                tokenized_document.append(word)  # Append it to a list
        tokenized.append(tokenized_document)         # Append list of words to a list
    return tokenized

texts = tokenize(documents)

texts               #  List of list. The inner list
                    #  contains tokens of respective documents


# 3.1 The following code is equivalent to above nested for-loops
# Nested list comprehension
[[word  for word in document.lower().split(' ') if word not in stoplist] for document in documents ]


# 4.
# Ref : https://www.ludovf.net/blog/python-collections-defaultdict/
#  A defaultdict is just like a regular Python dict,
#  except that it supports an additional argument at
#  initialization: a function. If someone attempts to
#  access a key to which no value has been assigned,
#  that function will be called (without arguments)
#  and its return value is used as the default value
#  for the key.

# 4.1 Initialise and create an empty dictionary
#     by name of 'frequency'
frequency = defaultdict(int)   # defaultdict(int) => key-values are int
                               # defaultdict(list) => key-values are lists
                               # Example: {'a' :['xx','yy'], 'b':['zz']}

# 4.2 Get count of each word in the 'documents'
for text in texts:
    for token in text:
    	# frequency[token] will first add a key 'token' to dict
    	#  (if the 'key' does not already exit) holding value '0'.
    	#   In either case value of the key will be incremented by 1
    	# So after all the loop is completed, value of each key
    	# will show its frequency
        frequency[token] += 1

frequency

# 4.3 Remove words that appear only once
#     So we create another list of lists
#     texts = [['he','he','to'],['to','go']]
#     frequency={'he' : 2, 'to': 2, 'go': 1}

output = list([])
for text in texts:
	tokens = list([])
	for token in text:
		if frequency[token] > 1:
			tokens.append(token)
	output.append(tokens)


# 4.4
print(output)      #     output = [['he','he','to'],['to']]

# 5. Module 'corpora.Dictionary' implements the concept of
#     Dictionary – a mapping between words and their integer ids.
#    Ref: https://radimrehurek.com/gensim/corpora/dictionary.html
dictionary = corpora.Dictionary(output)

# 5.1
dictionary      # Just informs where it is stroed in memory

# 5.2
print(dictionary.token2id)         # Another function is id2token

# 5.3 Convert document into the bag-of-words (bow)
#      format ie list of (integer-tokens, token_count) per document.
bag = [dictionary.doc2bow(text) for text in output]
bag

# 5.4 Just seperate integer-tokens from frequency
id_text=[]
for doc in bag:
    doclist=[]
    for id,_ in doc:
        doclist.append(id)
    id_text.append(doclist)

id_text

#####################################
