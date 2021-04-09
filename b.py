from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from random import randrange
import pandas as pd
from naive_bayes import *
import re
import math
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


def read_corpus(file_name):
    file = open(file_name, "r", encoding="utf8")
    rows = []
    for line in file:
        review = re.sub("[^a-zA-Z\s]", "", line.strip())
        review = review.split(" ", 3)
        genre = review[0].strip()
        _class = review[1].strip()
        _id = review[2].split("txt")[0]
        sentence = review[3]
        #print(genre, _class, _id, sentence)
        rows.append([genre, _class, _id, sentence])
    file.close()
    #print(rows)
    return rows;

"""
reviews = read_corpus("corpus2.txt")

a_string = "Th!is ?is a$ s@en!!te?!nce."
string_no_punctuation = re.sub("[^\w\s]", "", a_string)
word_list = string_no_punctuation.split(" ", 2)
#print(word_list)

txt = "ulkahf  şjşfa"
a = txt.split()
print(a) """

import nltk
text = "Hi, I want to get the bigram list of this string"
for item in nltk.bigrams(text.split()):
    print(' '.join(item))

vectorizer = CountVectorizer(ngram_range=(2, 2))
columns = ["genre", "class", "id", "sentence"]
reviews = read_corpus("corpus3.txt")



"""
sentences = []
for review in reviews:
    sentence = review[-1]
    for word in nltk.bigrams(sentence.split()):
        sentences.append(word) 
    
"""
#print(sentences)
sentences = [sentence[-1] for sentence in reviews]

vec = CountVectorizer(ngram_range=(2, 2), stop_words="english")
X = vec.fit_transform(sentences)
tdm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
word_list = vec.get_feature_names()
count_list = X.toarray().sum(axis=0) 

neg_BoW = dict(zip(word_list,count_list))
print(neg_BoW.keys())



s = reviews[0][-1]
new_sent = re.sub("[^\w\s]", "", s.strip())
bigrams = nltk.bigrams(new_sent.split())
clean = [word for word in bigrams if not any(stop in word for stop in stop_words)]
print(clean)
print("-----------------------------------------------------------")

for x in clean:
    x = " ".join(x)
    if x in neg_BoW.keys():
        print(x, "***")