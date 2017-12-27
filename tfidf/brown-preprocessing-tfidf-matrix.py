# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 12:56:09 2017

@author: Vlatka

Preprocessing Brown in 15 categories

import preprocessing

- create wordsall - words from Brown corpora
- remove punctuation, stopwords
- stemming (Porter stemmer)
- vectorizer - generate matrix of term frequency 
- vocabulary

MemoryError in freq_term_matrix.toarray() 
    -> divided corpora into files (category)
  
tf–idf (term frequency–inverse document frequency) is a numerical 
statistic that is intended to reflect how important a word is to a 
document in a collection or corpus.
2 ways to generate representation:
1. Generate a matrix of token/phrase counts from a collection of text 
   documents using CountVectorizer and feed it to TfidfTransformer 
   to generate the TF/IDF representation.

2. Feed the collection of text documents directly to TfidfVectorizer 
   and go straight to the TF/IDF representation skipping the middle man.

"""

from nltk.corpus import brown
from preprocessing_array import *
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import savetxt


    
with open("brown-vocabulary-stemm.txt", "r") as f: 
    vocabular= f.readlines()
f.close()
vocabular = [x.strip() for x in vocabular]

categories = brown.categories()
categories.remove('ca01')   # I don't know... something
categories.remove('ca02')

temp_doc=[]

for category in categories:
    fileids = brown.fileids(categories=category)
    
    doc_list=[]
    
    for ids in fileids:
        temp_doc = u' '.join(brown.words(fileids=ids))
        temp_doc = temp_doc.lower()
        doc_list.append(temp_doc)    

    tokens = [tokenize(words) for words in doc_list]
    tokens = remove_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)

    documents =[]
    for token in tokens:
        documents.append(' '.join(token))
        
    tfidf = TfidfVectorizer(vocabulary=vocabular)

    tfs = tfidf.fit_transform(documents)
    tfs_a = tfs.toarray()

    # will be removed
    with open("brown_"+category+"_tfidf_measures_matrix-voc.csv",'wb') as f:
        savetxt(f,vocabular,newline=' ', fmt="%s")
    f.close()
    with open("brown_"+category+"_tfidf_measures_matrix-voc.csv", 'ab') as f:
        savetxt(f, [''], fmt="%s", newline="\n", delimiter=",")
    f.close()   
    # 
    with open("brown_"+category+"_tfidf_measures_matrix-voc.csv", 'ab') as f:
        savetxt(f, tfs_a, delimiter=",")
    f.close()   

