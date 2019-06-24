import Preprocess
import features
import pandas as pd
from sklearn.linear_model import *
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score

# k = 3,8415 Voor StepAIC
qp = pd.read_csv('query_product2.csv')
qp2 = pd.read_csv('product_descriptions.csv')
qp = qp.join(qp2, on="product_uid" , lsuffix = "_dus")

def preprocess(qp):	
	
	qp = Preprocess.spellingcorrection(qp)
	Preprocess.removeBrackets(qp)
	qp = Preprocess.removeStopwords(qp)
	#pickle.dump(qp, open("qp.p", "wb"))

	return qp

#model1 = pickle.load(open("docs_666_mc1_size100_window7_duraton4207.481109857559.p", "rb"))
#qp = pickle.load(open("qp.p", "rb"))

dfcos = pickle.load(open("cos.p", "rb"))
dfeuc = pickle.load(open("euc.p", "rb"))
dfjac = pickle.load(open("jac.p", "rb"))
dfw2vp = pickle.load(open("w2v-withpreprocessing.p", "rb"))
dfw2v = pickle.load(open("w2v100-7.p", "rb"))

df = pd.concat([dfcos, dfeuc, dfjac, dfw2v, dfw2vp], axis=1,sort=False)

"""
df = pd.DataFrame(columns=['w2v-withpreprocessing'])
for i, (query, title, desc) in enumerate(zip(qp['search_term'], qp['product_title'],qp['product_description'])):
	#euc = features.euclidean_distance(query, title)
	#cos = features.cosine_similarity(query, title)
	#jac = features.jaccard_similarity(query, title)
	#jacdesc = features.jaccard_similarity(query, desc)
	w2v = features.w2v(query, title, model1)
	df.loc[i] = [w2v]
	if not i % 100:
		print(i)
"""

#pickle.dump(df, open("w2v-withpreprocessing.p", "wb"))


x_train = df[:50_000]
y_train = qp['relevance'][:50_000]
x_test = df[50_000:]
y_test = qp['relevance'][50_000:]
print(x_train)
print(y_train)
alg = LinearRegression()
alg.fit(x_train, y_train)



y_pred_test = alg.predict(x_test)
print(r2_score(y_test, y_pred_test))
fig, ax  = plt.subplots()
plt.scatter(y_test, y_pred_test)
plt.ylabel("Model Value")
plt.xlabel("World Value")
maxsize = int(max(ax.get_xlim()[1], ax.get_ylim()[1]))
ax.plot(range(0,maxsize), range(0,maxsize), ls="--", c=".3")


plt.show()