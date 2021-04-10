from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from random import randrange
import pandas as pd

import re
import math
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
"""
corpus = ["i", "am", "on", "a", "sugar", "crash", "crash"]
"""

vec = CountVectorizer(stop_words="english")
X = vec.fit_transform(corpus)
tdm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(tdm)

sentence = "is this document okay"

tfidf_vectorizer=TfidfVectorizer(use_idf=True, stop_words="english") 
 
# just send in all your docs here 
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(corpus)
# get the first vector out (for the first document) 
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
 
# place tf-idf values in a pandas data frame 
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df = df.sort_values(by=["tfidf"],ascending=False)
print(df.head(50))



""" vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names()) """
#print(X.get_feature_names())

