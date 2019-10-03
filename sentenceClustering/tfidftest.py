from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.corpus import stopwords
#import numpy as np

with open('stopwords_es.txt','r') as f:
	stopwords_es = f.read().split('\n')

with open('piñera_merge') as a:
	merge = [a.read()]

# Contando frecuencia
vectorizer = CountVectorizer(stop_words=stopwords_es)
X = vectorizer.fit_transform(merge)
sum_words = X.sum(axis=0) 
words_freq = [[word, sum_words[0, idx]] for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[0:20]
print('Frecuencia')
for word, freq in words_freq:
    print(word, freq)
	
# Haciendo TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords_es)
X = vectorizer.fit_transform(merge)
sum_words = X.sum(axis=0) 
words_freq = [[word, sum_words[0, idx]] for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[0:20]
print('TFIDF')
for word, freq in words_freq:
    print(word, freq)
	
# Usando N-gramas
vectorizer = CountVectorizer(stop_words=stopwords_es, ngram_range=(2,2))
X = vectorizer.fit_transform(merge)
sum_words = X.sum(axis=0) 
words_freq = [[word, sum_words[0, idx]] for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[0:20]
print('N-gramas')
for word, freq in words_freq:
    print(word, freq)