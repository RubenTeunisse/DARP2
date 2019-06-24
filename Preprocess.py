import nltk
from spellingcorrection import correction
import pandas as pd
import pickle
import re



def removeBrackets(qp):  
    qp["product_title"] = qp["product_title"].replace("\(.*\)", "", regex=True)
    return qp

def spellingcorrection(qp):
	dic = pickle.load( open( "correctiondict.p", "rb" ))
	for index, query in enumerate(qp['search_term']):
		if query in dic.keys():
			qp.loc[index, 'search_term'] = dic[query]

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
	stopwords = create_stopwords_set()    
	token_pattern = r"(?u)\b\w\w+\b"
	newqp = list()

	for index,title in enumerate(qp['product_title']):	
		tokens = [x.lower() for x in re.findall(token_pattern, title)]
		tokens = [x for x in tokens if x not in stopwords]
		tokens = " ".join(tokens)

		qp.loc[index, 'product_title'] =  tokens

		if index % 100 == 0:
			print(index)

	return qp
	#qp.loc['product_title'] = pd.DataFrame(newqp)

def stopwordtest(title):
	stopwords = create_stopwords_set()    
	token_pattern = r"(?u)\b\w\w+\b"

	tokens = [x.lower() for x in re.findall(token_pattern, title)]
	tokens = [x for x in tokens if x not in stopwords]
	tokens = " ".join(tokens)

	print(tokens)

stopwordtest("I am red rat rad hel rat hell hello no do by me as ma xi a b c d e f g h i j k l m n o p q r s t u v w x y z")
