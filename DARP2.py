#Vragen: L.P.Stoop@uu.nl
import numpy
import csv
import pandas
import re
import nltk
import word2vec.py

from tokenize import tokenize


qp = pandas.read_csv('query_product2.csv')


def removeBrackets(qp):  
    qp["product_title"] = qp["product_title"].replace("\(.*\)", "", regex=True)
    return qp

#print(removeBrackets(qp)['product_title'])


def get_term_frequency_matrix(doc):
    dic = {}
    tokenized = nltk.word_tokenize(doc)
    
    for i in range(0,len(tokenized)):
        if dic.has_key(tokenized[i]):
            dic[tokenized[i]] += 1
        else:
            dic[tokenized[i]] = 1
    
    return dic

def cosine_similarity(query, title):
    querytf = get_term_frequency_matrix(query)
    titletf = get_term_frequency_matrix(title)
    
    result = 0.0;
    
    for key in querytf.keys():
        if titletf.has_key(key):
            result += querytf[key] * titletf[key]
           
    query_magnitude = numpy.linalg.norm(querytf.values())
    title_magnitude = numpy.linalg.norm(titletf.values())
    
    denominator = query_magnitude * title_magnitude
    
    return result / denominator
    

def euclidean_distance(query,title):
    querytf = get_term_frequency_matrix(query)
    titletf = get_term_frequency_matrix(title)
    
    union = list(set(querytf.keys()) | set(titletf.keys()))
    result = 0.0
    
    for key in union:
        if not querytf.has_key(key):
            result += (titletf[key])**2
        elif not titletf.has_key(key):
            result += (querytf[key])**2
        else: result += (querytf[key] - titletf[key])**2
    
    return numpy.sqrt(result)

def create_stopwords_set():
    with open('stopwords.txt') as f:
        data = f.readlines()

    stopword_set = set()

    for line in data:
        if line != "":
            stopword_set.add(line[:-1])     #-1 to remove newline
    return stopword_set

def removeStopwords(input):
    stopwords = create_stopwords_set()
    tokenized_input = nltk.word_tokenize(input)
    output = list()    

    for i in range(0,len(tokenized_input)):
        if not tokenized_input[i] in stopwords:
            output.append(tokenized_input[i])

    return output

print(removeStopwords("hi my name is Niels and I really like this function"))



