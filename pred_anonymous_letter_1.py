import pandas as pd
import codecs
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from muffnn import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from muffnn import Autoencoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd


data = pd.read_csv("/Users/samweiss/downloads/cabinent_members_opening_testimony_semi_processed.csv")
data = pd.read_csv("/Users/samweiss/src/nytimes_resistance_letter/cab_member_testimony.csv", encoding='utf-8')

import re
re.sub('[^A-Za-z0-9]+', '', mystring)

from nltk import tokenize
sentences_by_member = [tokenize.sent_tokenize(str(x)) for x in data['Text']]
num_sentences_by_member = [len(x) for x in sentences_by_member]

flattened_sentences = [item for sublist in sentences_by_member for item in sublist]
flattened_sentences = [str(re.sub('\W+',' ', x)) for x in flattened_sentences]

y = np.repeat(data['Name'], num_sentences_by_member)
len(y) == len(flattened_sentences)
length_text = np.array([len(str(x)) for x in flattened_sentences])

vectorizer = CountVectorizer(analyzer='word', min_df=.005,lowercase = True, ngram_range = (1,2))
vectorizer.fit(flattened_sentences)
vectorized_text =  vectorizer.transform(flattened_sentences)


vectorizer_char = CountVectorizer(analyzer='char', min_df=.005,lowercase = True, ngram_range = (1,2))
vectorizer_char.fit(flattened_sentences)
vectorized_text_char =  vectorizer_char.transform(flattened_sentences)

from scipy.sparse import coo_matrix, hstack
vectorized_data = hstack([vectorized_text, vectorized_text_char])
letter_obs = np.where(y == 'letter')[0]
not_letter_obs = np.where(y != 'letter')[0]

y_training = y.iloc[not_letter_obs]
X_training = vectorized_data.todense()[not_letter_obs]

X_letter = vectorized_data.todense()[letter_obs]
scaler = StandardScaler()
X_std = scaler.fit_transform(X_training)

param_dist = {"C":  [.00001,.0001,.001,.01,.1,1,3,10,40,100,300,1000, 5000,10000,90000], "penalty":['l1','l2']}
clf = LogisticRegression(C=2,penalty = 'l2', class_weight='balanced')

gridsearch = GridSearchCV(clf, param_grid=param_dist)


gridsearch.fit(X_std, y_training)
preds = gridsearch.predict_proba(scaler.transform(X_letter))
#preds = gridsearch.predict_proba(X_letter)

avg_pred_cab_member = preds.mean(axis = 0)
avg_pred_cab_member = (preds>.1).mean(axis=0)
avg_pred_cab_member = preds.mean(axis = 0)

[print([gridsearch.classes_[x],avg_pred_cab_member[x]]) for x in np.argsort(avg_pred_cab_member)]

['sonny perdue', 0.042799391346238246]
['rick perry', 0.042827496150054725]
['kirstjen nielsen ', 0.04348531255720842]
['Kelly_John', 0.04388015088869095]
['gina haspel', 0.04408884887851884]
['mulvaney', 0.044449131042627626]
['mattis', 0.044520551404359426]
['zinke', 0.04460915299172605]
['mnuchin', 0.04475626096821514]
['mike pomeo', 0.04489294864505752]
['coats', 0.04504337727245659]
['wilbur ross', 0.0451681578002181]
['acosta', 0.045206788220377866]
['mcmahon', 0.04568208831113954]
['Robert Lighthizer', 0.04569583598035352]
['elaine chao', 0.04579825015621833]
['haley', 0.045990404889975765]
['devos', 0.04708158774824911]
['azar', 0.047256249688986826]
['ben carson', 0.04796853780460778]
['Robert L. Wilki', 0.0486808466773789]
['sessions', 0.05011863057734076]
