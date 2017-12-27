# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:07:58 2017

@author: Vlatka
"""

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re


def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def remove_punctuation(doc_list):
    word_list=[]
    punctuation1 = re.compile(r'[-.?!,":;()|0-9]')
    for worddoc in doc_list:
        words1 = [punctuation1.sub("", word) for word in worddoc]
        word_list.append(words1)
    return word_list

def remove_stopwords(doc_list):
    word_list=[]
    stop_words = set(stopwords.words('english'))
    for worddoc in doc_list:
        words = [w for w in worddoc if not w.lower() in stop_words]
        word_list.append(words)
    return word_list

def stemming(doc_list):
    word_list=[]
    porter_stemmer = PorterStemmer()
    for worddoc in doc_list:
        words = [porter_stemmer.stem(w) for w in worddoc]
        word_list.append(words)
    return word_list

