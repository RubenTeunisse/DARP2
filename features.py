import nltk
import numpy as np
import pickle

def w2v(query, title, model1):
    #print("model similarity for " + word1 + " and " + word2)    

    similarity_sum = 0.0
    no_comparisions = 1

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

def jaccard_similarity(string, query):    
    list1 = nltk.word_tokenize(string)
    try:
        list2 = nltk.word_tokenize(query)
    except:
        list2 = []
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def get_term_frequency_matrix(doc):
    dic = {}
    try: 
        tokenized = nltk.word_tokenize(doc)
    except:
        return dic
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
           
    query_magnitude = np.linalg.norm(list(querytf.values()))
    title_magnitude = np.linalg.norm(list(titletf.values()))
    
    if (title_magnitude != 0 and query_magnitude != 0):
        denominator = query_magnitude * title_magnitude
    else:
        denominator = 1;
    return result / denominator
    

def euclidean_distance(query,title):
    querytf = get_term_frequency_matrix(query)
    titletf = get_term_frequency_matrix(title)
    
    union = list(set(querytf.keys()) | set(titletf.keys()))
    result = 0.0
    
    for key in union:
        if not key in querytf:
            result += (titletf[key])**2
        elif not key in titletf:
            result += (querytf[key])**2
        else: result += (querytf[key] - titletf[key])**2
    
    return np.sqrt(result)