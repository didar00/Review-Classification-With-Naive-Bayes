from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

columns = ['sent', 'class']
rows = []

rows = [['This is my book', 'stmt'], 
        ['They are novels', 'stmt'],
        ['have you read this book', 'question'],
        ['who is the author', 'question'],
        ['what are the characters', 'question'],
        ['This is how I bought the book', 'stmt'],
        ['I like fictions', 'stmt'],
        ['what is your favorite book', 'question']]

training_data = pd.DataFrame(rows, columns=columns)

stmt_docs = [row['sent'] for index,row in training_data.iterrows() if row['class'] == 'stmt']

vec_s = CountVectorizer()
X_s = vec_s.fit_transform(stmt_docs)
print("X_s = ", X_s.toarray())
tdm_s = pd.DataFrame(X_s.toarray(), columns=vec_s.get_feature_names())
print(tdm_s)
print("----------------------------")

q_docs = [row['sent'] for index,row in training_data.iterrows() if row['class'] == 'question']

vec_q = CountVectorizer()
X_q = vec_q.fit_transform(q_docs)
tdm_q = pd.DataFrame(X_q.toarray(), columns=vec_q.get_feature_names())
print(tdm_q)
print("----------------------------")

word_list_s = vec_s.get_feature_names();
print("feature names = ", word_list_s) 
count_list_s = X_s.toarray().sum(axis=0)
print("count_list_s = ", count_list_s)
freq_s = dict(zip(word_list_s,count_list_s))
print(freq_s)

word_list_q = vec_q.get_feature_names();    
count_list_q = X_q.toarray().sum(axis=0) 
freq_q = dict(zip(word_list_q,count_list_q))