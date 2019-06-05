import numpy
import csv
import pandas
import re
import nltk
from tokenize import tokenize
from nltk.corpus import wordnet
synonyms = []
antonyms = []

for syn in wordnet.synsets("active"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			 antonyms.append(l.antonyms()[0].name())

print(type(synonyms[0].str()))
print(set(antonyms))


qp = pandas.read_csv('query_product.csv')

def removeBrackets(qp):  
    qp["product_title"] = qp["product_title"].replace("\(.*\)", "", regex=True)
    return qp


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

#print(euclidean_distance("komt komt gat hoi","komt mehha hoi"))



