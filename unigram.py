from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.corpus import stopwords
from random import randrange
import pandas as pd
import numpy as np
import math
import nltk
import re


# import stop words from nltk
stop_words = set(stopwords.words("english"))


"""
Reads corpus line by line with cleaning data
(removing non-alphabetical characters, punctuations etc.)
"""
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
        rows.append([genre, _class, _id, sentence])
    file.close()
    return rows;


"""
Splits the data into two parts: test and training
Training: %80; Test %20
"""
def train_test_split(reviews, test_percent):
    reviews_copy = list(reviews)
    test_size = int(len(reviews)*(test_percent))
    train_list = list()
    test_list = list()
    for _ in range(test_size):
        index = randrange(len(reviews_copy))
        test_list.append(reviews_copy.pop(index))
    train_list = reviews_copy
    train_test_list = [train_list, test_list]
    return train_test_list


"""
Naive Bayes Classifier with laplace
smooting to avoid zero frequency
probabilites
"""
def naive_bayes_with_laplace(test_sentence, BoW, total_word_counts, total_unique_words):
    sentence = get_words(test_sentence)
    cond_prob = 0
    for word in sentence:
        if word in BoW.keys():
            count = BoW[word]
        else:
            count = 0
        cond_prob += math.log((count + 1)/(total_word_counts + total_unique_words))
    return cond_prob


"""
Returns cleaned version of the sentence
in the form of words
"""
def get_words(sentence):
    new_sent = re.sub("[^a-zA-Z\s]", "", sentence.strip())
    words = new_sent.split()
    # remove stop words
    words = [word for word in words if word not in stop_words]
    return words


"""
Calculates the probability of classes with given
sentences
"""
def classify_sentence(sentence):
    pos_review_likelihood = naive_bayes_with_laplace(sentence, pos_BoW, total_counts_words_pos, total_unique_words)
    neg_review_likelihood = naive_bayes_with_laplace(sentence, neg_BoW, total_counts_words_neg, total_unique_words)

    pos_sents = len(pos_sentences)
    neg_sents = len(neg_sentences)

    # prior probability
    pos_prior = pos_sents/(pos_sents+neg_sents)
    neg_prior = neg_sents/(pos_sents+neg_sents)

    # posterior probability with log
    pos_prob = pos_review_likelihood + math.log(pos_prior)
    neg_prob = neg_review_likelihood + math.log(neg_prior)

    return "pos" if pos_prob > neg_prob else "neg"


"""
Classifies given test dataset,
stores each prediction with their
actual value in a list and returns
that list
"""
def classify(test_list):
    predictions = list()
    for review in test_list:
        prediction = classify_sentence(review[-1])
        predictions.append((prediction, review[1]))
    return predictions


"""
Given prediction/actual values
list, calculates the accuracy of the model
"""
def accuracy(predictions):
    true_preds = 0
    false_preds = 0
    for pred, actual in predictions:
        if pred == actual:
            true_preds += 1
        else:
            false_preds += 1
    return true_preds/(true_preds+false_preds)

"""
Returns the top N frequent words in
Bag of Words
"""
def top_N_words(BoW, N):
    print(sorted(BoW, key=BoW.get, reverse=True)[:N])


def print_review_split(review_split):
    print("------------------TRAIN DATA--------------------")
    for review in ttl[0]:
        print(review)
    print("------------------TEST DATA--------------------")
    for review in ttl[1]:
        print(review)



columns = ["genre", "class", "id", "sentence"]
# read the corpus
reviews = read_corpus("all_sentiment_shuffled.txt")
ttl = train_test_split(reviews, 0.2) # train, test data split list

# stores the training data in dataframe with given column names
training_data = pd.DataFrame(ttl[0], columns=columns)

# get positive sentences
pos_sentences = [row["sentence"] for index,row in training_data.iterrows() if row["class"] == "pos"]
vec_pos = CountVectorizer(stop_words="english")
X_pos = vec_pos.fit_transform(pos_sentences)

#get negative sentences
neg_sentences = [row["sentence"] for index, row in training_data.iterrows() if row["class"] == "neg"]
vec_neg = CountVectorizer(stop_words="english")
X_neg = vec_neg.fit_transform(neg_sentences)


"""
BAG OF WORDS METHOD IMPLEMENTED WITH DICTIONARIES
"""

# each unique word in positive reviews
word_list_pos = vec_pos.get_feature_names() 
# number of occurences of each word in positive reviews
count_list_pos = X_pos.toarray().sum(axis=0) 
# bag of words for positive reviews
# which contains each word in positive
# reviews with their relevant frequencies
# in that class
pos_BoW = dict(zip(word_list_pos,count_list_pos)) 


# each unique word in negative reviews
word_list_neg = vec_neg.get_feature_names() 
# number of occurences of each word in negative reviews
count_list_neg = X_neg.toarray().sum(axis=0) 
# bag of words for negative reviews
# which contains each word in negative
# reviews with their relevant frequencies
# in that class
neg_BoW = dict(zip(word_list_neg,count_list_neg))



# list of reviews from the corpus to get the vocabulary features
sentences = [row["sentence"] for index,row in training_data.iterrows()]

vec = CountVectorizer(stop_words="english")
X = vec.fit_transform(sentences)

# number of unique words that is in the reviews
total_unique_words = len(vec.get_feature_names())

# total number of words that is in the positive reviews
total_counts_words_pos = count_list_pos.sum(axis=0)
# total number of words that is in the negative reviews
total_counts_words_neg = count_list_neg.sum(axis=0)


# classify the test dataset
predictions = classify(ttl[1])
print(accuracy(predictions))