from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from random import randrange
import pandas as pd
from naive_bayes import *
import re
import math

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def read_corpus(file_name):
    file = open(file_name, "r", encoding="utf8")
    rows = []
    for line in file:
        review = re.sub("[^\w\s]", "", line.strip())
        review = review.split(" ", 3)
        genre = review[0].strip()
        _class = review[1].strip()
        _id = review[2].split("txt")[0]
        sentence = review[3]
        rows.append([genre, _class, _id, sentence])
    file.close()
    #print(rows)
    return rows;
        

def train_test_split(reviews, test_percent):
    reviews_copy = list(reviews)
    #print(len(reviews))
    test_size = int(len(reviews)*(test_percent))
    train_list = list()
    test_list = list()
    for _ in range(test_size):
        index = randrange(len(reviews_copy))
        test_list.append(reviews_copy.pop(index))
    train_list = reviews_copy
    train_test_list = [train_list, test_list]
    return train_test_list


def naive_bayes_with_laplace(test_sentence, BoW, total_word_counts, total_unique_words):
    sentence = get_words(test_sentence)
    #print(sentence)
    #print("----------------")
    cond_prob = 0 # ATTTTENNNNTTTIIIIIOOONNNNN??!!!!!
    for word in sentence:
        #print(word)
        if word in BoW.keys():
            count = BoW[word]
            #print("count :", BoW[word])
        else:
            count = 0
        cond_prob += math.log((count + 1)/(total_word_counts + total_unique_words))
    return cond_prob


def get_words(sentence):
    new_sent = re.sub("[^\w\s]", "", sentence.strip())
    words = new_sent.split()
    words = [word for word in words if word not in stop_words]
    return words



def classify_sentence(sentence):
    pos_review_likelihood = naive_bayes_with_laplace(sentence, pos_BoW, total_counts_words_pos, total_unique_words)
    neg_review_likelihood = naive_bayes_with_laplace(sentence, neg_BoW, total_counts_words_neg, total_unique_words)
    #print("pos: ", pos_review_likelihood, "neg ", neg_review_likelihood)


    pos_sents = len(pos_sentences) ## move this part make it global
    neg_sents = len(neg_sentences) ## rather than repeating it  '''LATER'''
    #print("pos: ", pos_sents, " neg: ", neg_sents)
    pos_prior = pos_sents/(pos_sents+neg_sents)
    neg_prior = neg_sents/(pos_sents+neg_sents)

    pos_prob = math.log(pos_review_likelihood) + math.log(pos_prior)
    neg_prob = math.log(neg_review_likelihood) + math.log(neg_prior)
    #print("positive : ", pos_prob, " negative : ", neg_prob)
    #print()

    return "pos" if pos_prob > neg_prob else "neg"


def classify(test_list):
    predictions = list()
    for review in test_list:
        #print()
        #print(review[-1])
        prediction = classify_sentence(review[-1])
        predictions.append((prediction, review[1]))
    return predictions


def accuracy(predictions):
    true_preds = 0
    false_preds = 0
    for pred, actual in predictions:
        if pred == actual:
            true_preds += 1
        else:
            false_preds += 1
    return true_preds/(true_preds+false_preds)


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
reviews = read_corpus("all_sentiment_shuffled.txt")
ttl = train_test_split(reviews, 0.2)
#print_review_split(ttl)

training_data = pd.DataFrame(ttl[0], columns=columns)
#print(training_data)

pos_sentences = [row["sentence"] for index,row in training_data.iterrows() if row["class"] == "pos"]
#print(pos_sentences)
vec_pos = CountVectorizer(stop_words="english")
X_pos = vec_pos.fit_transform(pos_sentences)
tdm_s = pd.DataFrame(X_pos.toarray(), columns=vec_pos.get_feature_names())

neg_sentences = [row["sentence"] for index, row in training_data.iterrows() if row["class"] == "neg"]
vec_neg = CountVectorizer(stop_words="english")
X_neg = vec_neg.fit_transform(neg_sentences)
tdm_s = pd.DataFrame(X_neg.toarray(), columns=vec_neg.get_feature_names())

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
#print(pos_BoW)

# each unique word in negative reviews
word_list_neg = vec_neg.get_feature_names() 
# number of occurences of each word in negative reviews
count_list_neg = X_neg.toarray().sum(axis=0) 
# bag of words for negative reviews
# which contains each word in negative
# reviews with their relevant frequencies
# in that class
neg_BoW = dict(zip(word_list_neg,count_list_neg))
#print(neg_BoW["appointed"])

#top_N_words(pos_BoW, 10)
#top_N_words(neg_BoW, 10)


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(X_pos)
# print idf values 
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=vec_pos.get_feature_names(),columns=["idf_weights"]) 
 
# sort ascending 
df_idf.sort_values(by=['idf_weights'])

#print(df_idf.tail(50))

sentences = [row["sentence"] for index,row in training_data.iterrows()]

vec = CountVectorizer(stop_words="english")
X = vec.fit_transform(sentences)

# number of unique words that is in the reviews
total_unique_words = len(vec.get_feature_names())
#print(total_unique_words)

# total number of words that is in the positive reviews
total_counts_words_pos = count_list_pos.sum(axis=0)
#print(total_counts_words_pos)
# total number of words that is in the negative reviews
total_counts_words_neg = count_list_neg.sum(axis=0)
#print(total_counts_words_neg)

#print(ttl[1])
predictions = classify(ttl[1])
print(accuracy(predictions))