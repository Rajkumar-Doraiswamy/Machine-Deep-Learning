# -*- coding: utf-8 -*-
"""
Last amended: 27th April, 2018
Folder: C:\Users\ashok\OneDrive\Documents\bagOfWordsMeetsBagofPopcorn\smallexpt
		/home/ashok/Documents/9.word2vec

Kaggle:
    https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors
Analytics Vidhya:    
    https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
    
Objective:
    Experimentation with pre-created word2vec file
    
"""

# 1. Reset variables. Import module to manipulate word2vec files
%reset -f
import  gensim.models 
import os

# 2. Google word2vec file is in the following folder
#    File size is around 4 gb
#     We will not use this file
# os.chdir("E:\\googleWord2Vec")

# 3. Kaggle's word2vec file folder
# os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\bagOfWordsMeetsBagofPopcorn/smallexpt")
os.chdir("/home/ashok/Documents/9.word2vec")

# 4. Load the word2vec file
#     Refer bagofpopcorn.py. This file was created by Ashok K Harnal 
#      File size is around 59MB
model =   gensim.models.keyedvectors.KeyedVectors.load('/home/ashok/.keras/datasets/bagOfPopcorn/300features_40minwords_10context')

#### Testing Word to Vector model

# 5. What does the vector look like?
dog = model['dog']
dog.shape
dog
# 6.
model.doesnt_match("man woman child kitchen".split())
# 7.
model.doesnt_match("man woman child cow".split())
# 8
model.doesnt_match("france england germany berlin".split())
# 9.
model.doesnt_match("paris berlin london austria".split())
# 10.
model.most_similar("man")  
# 11.
model.most_similar("queen")
# 12.
model.most_similar("awful")
model.most_similar("bank")     # 'bank' has two meanings. River bank. Finance bank.
# 13. Maths with word2vec model. Two vectors get added and one subtracted.
# King – Man + Woman = ?
model.most_similar(positive=['woman', 'king'], negative=['man'])
model.most_similar(positive=['woman', 'uncle'], negative=['man'])
model.most_similar(positive=[ 'king'], negative=['woman'])

#########################