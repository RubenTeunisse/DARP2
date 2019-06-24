# Code obtained from https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Python program to generate word vectors using Word2Vec 

# importing all necessary modules 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import pandas
import pickle
import time
import gensim 
from gensim.models import Word2Vec 

def create_titles_pickle():
	pd = pandas.read_csv('query_product2.csv')['product_title']
	pickle.dump(pd, open("product_titles.p", "wb"))
	print("done")

def create_description_pickle():
	pd = pandas.read_csv('product_descriptions.csv')['product_description']
	pickle.dump(pd, open("descriptions.p", "wb"))

def train_word2vec_descriptions(no_docs, mincount = 1, grootte = 100, raam = 5):
	
	descriptions = pickle.load(open("descriptions.p", "rb"))
	pd = descriptions[:no_docs]

	data = []

	# iterate through each sentence in the file
	count = 1 
	for doc in pd:
		if count % 1000 == 0:
			print("doc " + str(count))
		count += 1
		for i in sent_tokenize(doc): 
			temp = [] 

			# tokenize the sentence into words 
			for j in word_tokenize(i): 
				temp.append(j.lower()) 

			data.append(temp) 

	print("train CBOW model. no_docs: " + str(no_docs) + " size: " + str(grootte) + " window: " + str(raam))

	# Create CBOW model - Continuous bag of words 
	model1 = gensim.models.Word2Vec(data, min_count = mincount, 
								size = grootte, window = raam) 

	return(model1)

def train_word2vec_titles(no_docs, mincount = 1, grootte = 100, raam = 5):

	pd = pickle.load(open("product_titles.p", "rb"))

	data = []

	# iterate through each sentence in the file
	count = 1 
	for doc in pd:
		if count % 250 == 0:
			print("doc " + str(count))
		count += 1
		for i in sent_tokenize(doc): 
			temp = [] 

			# tokenize the sentence into words 
			for j in word_tokenize(i): 
				temp.append(j.lower()) 

			data.append(temp) 

	print("train CBOW model. no_docs: " + str(no_docs) + " size: " + str(grootte) + " window: " + str(raam))

	# Create CBOW model - Continuous bag of words 
	model1 = gensim.models.Word2Vec(data, min_count = mincount, 
								size = grootte, window = raam) 

	return(model1)

def apply_word2vec(query, title, modelpath):
	print("model similarity for " + word1 + " and " + word2)
	model1 = pickle.load(open(modelpath, "rb"))

	similarity_sum = 0.0
	no_comparisions = 0.0

	for queryword in query:
		for titleword in title:
			no_comparisions += 1
			try:			
				similarity_sum += model1.similarity(queryword,titleword)
			except:
				similarity_sum += 0
				no_comparisions -= 1

	avg_similarity = similarity_sum / no_comparisions

	return(avg_similarity)
	

def stopwatch(start):
	duration = time.time() - start
	print(duration)

	return duration
def train_multiple_models(no_docs):

	### size 50 100 150.  Window 5. ###
	print("### SIZE ###")
	"""
	train_single_model(no_docs, 50, 5)
	train_single_model(no_docs, 100, 5)
	train_single_model(no_docs, 150, 5)	
	"""
	#####################################


	### window 3 5 7 9.  size 100. . ###
	print()
	print("### WINDOW ###")
	#train_single_model(no_docs, 100, 3)
	train_single_model(no_docs, 100, 7)
	train_single_model(no_docs, 100, 9)
	#####################################

def train_single_model(no_docs, size, window):
	path = "word2vec models\\"

	start = time.time()
	model = train_word2vec_titles(no_docs, 1, size, window)
	duration = stopwatch(start)
	pickle.dump(model, open(path + "docs_" + str(no_docs) + "_mc1_size" + str(size) + "_window" + str(window) + "_duraton" + str(duration) + ".p", "wb"))	

train_multiple_models(666)

