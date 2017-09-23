import pandas as pd
import numpy as np
import pickle
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from string import punctuation
import math
import re
import matplotlib.pyplot as plt
ps = PorterStemmer()


def save_result(obj, filepath):
    with open(filepath, 'wb') as out:
        pickle.dump(obj, out)


def filter_text(text):

    filter_words = stopwords.words('english')+list(punctuation)
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in filter_words]
    filtered_sentence = ' '.join(filtered_sentence)
    #filter_text_no_numbers = filtered_sentence
    filter_text_no_numbers = re.sub("\d+",'', filtered_sentence)
    return filter_text_no_numbers

def stemming(text):
    words = word_tokenize(text)
    stem_words=[]
    for w in words:
        stem_words = stem_words+[(ps.stem(w))]
    stem_sent = ' '.join(stem_words)
    return stem_sent


def tfidf(folderpath):
    docs = glob(folderpath)
    doc_string=[]
    for doc in docs:
        text = open(doc,'r')
        lower_text = ''.join(text.readlines()).lower()
        doc_string.append(stemming(filter_text(lower_text)))
    tfidf = TfidfVectorizer()
    tfidf_matrix = pd.DataFrame(tfidf.fit_transform(doc_string).todense())
    return tfidf_matrix

movie_review_df = pd.read_csv("\\Users\\mpdur\\Downloads\\movie_reviews.csv",header=None)
movie_review_df.columns= ['Rating', 'Review']
for movie in movie_review_df.iterrows():
    review = ((movie[1]['Review']))
    file = open('\\Users\\mpdur\\Downloads\\IDS566HW2\\{movie}.txt'.format(movie= movie[0]),'w')
    file.write(review)
    file.close()

tfidf_matrix = tfidf('\\Users\\mpdur\\Downloads\\IDS566HW2\\*.txt')

def mse(y,yhat):
    sq_err =0
    for i in range(len(y)):
        sq_err = sq_err+(y[i]-yhat[i])**2
    return(sq_err/len(y))

def diff_mse(y,yhat):
    err = 0
    for i in range(len(y)):
        err = err+(y[i]-yhat[i])
    return(err/2)

def linear_regression(X, y, lr=0.05, n_epoch=200):
    mse_array=[]
    curr_x = [0]*X.shape[1]
    pre_x = [0]*X.shape[1]
    for i in range(n_epoch):
        yhat = np.dot(tfidf_matrix,curr_x)
		#applying gradient descent equation
        curr_x = pre_x - (lr)*diff_mse(yhat,y)
        print("Iteration:{i}; MSE:{mse}".format(i=i,mse=mse(y,yhat)))
        mse_array.append(mse(y,yhat))
    plt.plot(mse_array)
    plt.ylabel('MSE')
    plt.savefig('\\Users\\mpdur\\Downloads\\671192091.png')
    plt.show()
    return yhat

linear_regression(tfidf_matrix,movie_review_df.Rating, lr = 0.05, n_epoch= 200)
