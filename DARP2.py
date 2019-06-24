#Vragen: L.P.Stoop@uu.nl
import numpy
import csv
import pandas
import re
import nltk
import pickle
import sklearn

from tokenize import tokenize


qp = pandas.read_csv('query_product2.csv')


def removeBrackets(qp):  
    qp["product_title"] = qp["product_title"].replace("\(.*\)", "", regex=True)
    return qp

def create_stopwords_set():
    with open('stopwords.txt') as f:
        data = f.readlines()

    stopword_set = set()

    for line in data:
        if line != "":
            stopword_set.add(line[:-1])     #-1 to remove newline
    return stopword_set

def removeStopwords(qp):
    product_title = qp["product_title"]
    stopwords = create_stopwords_set()
    tokenized_input = nltk.word_tokenize(product_title)
    output = list()    

    for i in range(0,len(tokenized_input)):
        if not tokenized_input[i] in stopwords:
            output.append(tokenized_input[i])

    return output

print(removeStopwords("My own laptop is a lkasjdf;ljkasdf"))

def spellingcorrection():
    dic = pickle.load( open( "correctiondict.p", "rb" ))
    for query in qp['search_term']:
        if query in dic.keys():
            query = dic[query]


def get_term_frequency_matrix(doc):
    dic = {}
    tokenized = nltk.word_tokenize(doc)
    
    for i in range(0,len(tokenized)):
        if tokenized[i] in dic:
            dic[tokenized[i]] += 1
        else:
            dic[tokenized[i]] = 1
    
    return dic

def cosine_similarity(query, title):
    querytf = get_term_frequency_matrix(query)
    titletf = get_term_frequency_matrix(title)
    
    result = 0.0;
    
    for key in querytf.keys():
        if key in titletf:
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

#print(removeStopwords("hi my name is Niels and I really like this function"))

def preprocess(qp):
    spellingcorrection()
    no_brackets = removeBrackets(qp)
    #no_stopwords = removeStopwords(no_brackets)
    pickle.dump(no_brackets, open("preprocessed.p", "wb"))   

    return no_brackets

def create_feature_csvs(voorbewerkt):
    cos_similarities = "cosine_similarity,relevance,"
    euc_distances = "euclidean_distance,relevance,"
    print(voorbewerkt['search_term'][2])
    for i in range(1, len(voorbewerkt) - 1):     
        print(voorbewerkt['search_term'][i])
        cos_similarities += str(cosine_similarity(voorbewerkt['search_term'][i], voorbewerkt['product_title'][i])) + "," + str(voorbewerkt['relevance'][i]) + ","        
        euc_distances += str(euclidean_distance(row[3], row[2])) + "," + str(row[4]) + ","

    pickle.dump(cos_similarities[:-1], open("cos_similarities.csv", "wb"))
    pickle.dump(euc_distances[:-1], open("euc_distances.csv", "wb"))


#voorbewerkt = preprocess(qp)
#create_feature_csvs(qp)