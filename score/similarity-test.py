# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:46:44 2017

@author: Vlatka
"""

from __future__ import division
import numpy as np
import math
from decimal import *
from os import listdir
from os.path import isfile, join
from numpy import savetxt

def sum_of_squares(vector):
    sum = 0
    for vx in vector:
        sum += Decimal(vx)*Decimal(vx)
    return sum

def score(query_vector,document_vector):
    sum = 0
    for vx,vy in zip(query_vector,document_vector):
        sum += float(vx)*float(vy)
    
    return (sum/(math.sqrt(sum_of_squares(query_vector) * sum_of_squares(document_vector))))

# test one sample:
query = np.genfromtxt("adventure-vector.csv", delimiter=",",names=None, dtype=Decimal)
print query

mypath="training-documents"
outpath="score-documents"
training_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in training_files:
    print file
    documents_vector = np.genfromtxt(mypath+"/"+file, delimiter=",", names=None, dtype=Decimal)
    print documents_vector

    print "----documents similarity ----"
    scores=[]
    for document in documents_vector:
        scores.append(score(query,document))
    with open(outpath+"/score_1_"+file,'wb') as f:
        savetxt(f, scores, fmt="%f")
    f.close()

