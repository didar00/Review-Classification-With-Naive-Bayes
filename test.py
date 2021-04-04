from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
#from nltk.tokenize import word_tokenize


columns = ['sent', 'class']

def readCorpus(file_name):
    file = open("corpus.txt", "r")
    rows = []
    for line in file_name:
        rows.append(line.strip().split(", "))
    file.close()
    return rows;
    #print(rows)

rows = readCorpus("corpus.txt")
training_data = pd.DataFrame(rows, columns=columns)

stmt_docs = [row['sent'] for index,row in training_data.iterrows() if row['class'] == 'stmt']
vec_s = CountVectorizer()
X_s = vec_s.fit_transform(stmt_docs)
tdm_s = pd.DataFrame(X_s.toarray(), columns=vec_s.get_feature_names())

q_docs = [row['sent'] for index,row in training_data.iterrows() if row['class'] == 'question']
vec_q = CountVectorizer()
X_q = vec_q.fit_transform(q_docs)
tdm_q = pd.DataFrame(X_q.toarray(), columns=vec_q.get_feature_names())

word_list_s = vec_s.get_feature_names();
count_list_s = X_s.toarray().sum(axis=0)
freq_s = dict(zip(word_list_s,count_list_s))

word_list_q = vec_q.get_feature_names();    
count_list_q = X_q.toarray().sum(axis=0) 
freq_q = dict(zip(word_list_q,count_list_q))


docs = [row['sent'] for index,row in training_data.iterrows()]

vec = CountVectorizer()
X = vec.fit_transform(docs)

total_features = len(vec.get_feature_names())

total_cnts_features_s = count_list_s.sum(axis=0)
total_cnts_features_q = count_list_q.sum(axis=0)

new_sentence = 'what is the price of the book'
new_word_list = new_sentence.split(" ")


'''
with laplace
'''

prob_s_with_ls = []
for word in new_word_list:
    if word in freq_s.keys():
        count = freq_s[word]
    else:
        count = 0
    prob_s_with_ls.append((count + 1)/(total_cnts_features_s + total_features))
sent_probs = dict(zip(new_word_list,prob_s_with_ls))


prob_q_with_ls = []
for word in new_word_list:
    if word in freq_q.keys():
        count = freq_q[word]
    else:
        count = 0
    prob_q_with_ls.append((count + 1)/(total_cnts_features_q + total_features))
quest_probs = dict(zip(new_word_list,prob_q_with_ls))

prob_sent = 0.5 # might cause a problem later
for word in sent_probs.keys():
    print(word, " " , sent_probs[word])
    prob_sent *= sent_probs[word]
print(prob_sent)

prob_quest = 0.5 # might cause a problem later
for word in quest_probs.keys():
    print(word, " " , quest_probs[word])
    prob_quest *= quest_probs[word]
print(prob_quest)