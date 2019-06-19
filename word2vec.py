# Code obtained from https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Python program to generate word vectors using Word2Vec 

# importing all necessary modules 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import pandas
import pickle

warnings.filterwarnings(action = 'ignore') 

import gensim 
from gensim.models import Word2Vec 

def train_word2vec():
	
	pd = pandas.read_csv('product_descriptions.csv')['product_description'][:500]

	data = []

	# iterate through each sentence in the file
	count = 1 
	for doc in pd:
		print("doc " + str(count))
		count += 1
		for i in sent_tokenize(doc): 
			temp = [] 

			# tokenize the sentence into words 
			for j in word_tokenize(i): 
				temp.append(j.lower()) 

			data.append(temp) 

	print("train CBOW model")
	# Create CBOW model - Continuous bag of words 
	model1 = gensim.models.Word2Vec(data, min_count = 1, 
								size = 100, window = 5) 
	"""print("train skip gram model")
	# Create Skip Gram model 
	model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
												window = 5, sg = 1) 
												"""
	print("done")
	
	pickle.dump(model1, open("model1.p", "wb"))

	return(model1)

def apply_word2vec(word1, word2):
	print("apply model1")
	model1 = pickle.load(open("model1.p", "rb"))

	model1_sim = model1.similarity(word1,word2)
	"""
	print("apply model2")
	model2_sim = model2.similarity(word1,word2)
	"""
	return(model1_sim)
	#return (model1_sim,model2_sim)

(CBOWmodel) = train_word2vec()   #  "This is sentence 1. This is sentece 2. Is there a pattern between these sentences?")

print(apply_word2vec("they", "also"))