from sklearn.feature_extraction.text import CountVectorizer
from random import randrange
import pandas as pd

class Naive_Bayes:

    def __init__(self, ttl, ):
        self.ttl = ttl
        self.pos_BoW = pos_BoW
        self.neg_BoW = neg_BoW
        self.pos_sentences = pos_sentences
        self.neg_sentences = neg_sentences
        self.total_unique_words = total_unique_words
        self.total_counts_words_pos = total_counts_words_pos
        self.total_counts_words_neg = total_counts_words_neg


    def read_corpus(self, file_name):
        file = open(file_name, "r", encoding="utf8")
        rows = []
        for line in file:
            review = line.strip().split(" ", 3)
            genre = review[0]
            _class = review[1]
            _id = review[2]
            sentence = review[3]
            rows.append([genre, _class, _id, sentence])
        file.close()
        #print(rows)
        return rows;
        

    def train_test_split(self, reviews, test_percent):
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


    def naive_bayes_with_laplace(self, test_sentence, BoW, total_word_counts, total_unique_words):
        cond_prob = 1
        for word in test_sentence:
            if word in BoW.keys():
                count = BoW[word]
            else:
                count = 0
            cond_prob *= (count + 1)/(total_word_counts + total_unique_words)
        return cond_prob


    def classify_sentence(self, sentence):
        pos_review_likelihood = naive_bayes_with_laplace(sentence, pos_BoW, total_counts_words_pos, total_unique_words)
        neg_review_likelihood = naive_bayes_with_laplace(sentence, neg_BoW, total_counts_words_neg, total_unique_words)

        pos_sents = len(pos_sentences)
        neg_sents = len(neg_sentences)
        pos_prior = pos_sents/(pos_sents+neg_sents)
        neg_prior = neg_sents/(pos_sents+neg_sents)

        pos_prob = pos_review_likelihood*pos_prior
        neg_prob = neg_review_likelihood*neg_prior

        return "pos" if pos_prob > neg_prob else "neg"


    def classify(self, test_list):
        predictions = list()
        for review in test_list:
            prediction = classify_sentence(review[-1])
            predictions.append((prediction, review[1]))
        return predictions


    def accuracy(self, predictions):
        true_preds = 0
        false_preds = 0
        for pred, actual in predictions:
            if pred == actual:
                true_preds += 1
            else:
                false_preds += 1
        return true_preds/(true_preds+false_preds)


    def print_review_split(self, review_split):
        print("------------------TRAIN DATA--------------------")
        for review in ttl[0]:
            print(review)
        print("------------------TEST DATA--------------------")
        for review in ttl[1]:
            print(review)




