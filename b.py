import numpy as np
import re

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
        #print(genre, _class, _id, sentence)
        rows.append([genre, _class, _id, sentence])
    file.close()
    #print(rows)
    return rows;


reviews = read_corpus("corpus2.txt")

a_string = "Th!is ?is a$ s@en!!te?!nce."
string_no_punctuation = re.sub("[^\w\s]", "", a_string)
word_list = string_no_punctuation.split(" ", 2)
#print(word_list)

txt = "ulkahf  şjşfa"
a = txt.split()
print(a)