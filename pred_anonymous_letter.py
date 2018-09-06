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


data = pd.read_csv("/cabinent_members_opening_testimony_semi_processed.csv")

length_text = np.array([len(str(x)) for x in data['raw_text']])
to_keep = np.where(length_text > 1)[0]
data = data.iloc[to_keep]
data['raw_text'] = np.array([str(x) for x in data['raw_text']])
vectorizer = CountVectorizer(analyzer='word', min_df=.005,lowercase = True, ngram_range = (1,2))
vectorizer.fit(data['raw_text'])
vectorized_text =  vectorizer.transform(data['raw_text'])


vectorizer_char = CountVectorizer(analyzer='char', min_df=.005,lowercase = True, ngram_range = (1,2))
vectorizer_char.fit(data['raw_text'])
vectorized_text_char =  vectorizer_char.transform(data['raw_text'])

from scipy.sparse import coo_matrix, hstack
vectorized_data = hstack([vectorized_text, vectorized_text_char])
vectorized_data.to()
letter_obs = np.where(data['y'] == 'letter')[0]
not_letter_obs = np.where(data['y'] != 'letter')[0]

y_training = data['y'].iloc[not_letter_obs]
X_training = vectorized_data.todense()[not_letter_obs]

X_letter = vectorized_data.todense()[letter_obs]



param_dist = {"C":  sp_randint(.1, 100)}
clf = LogisticRegression(C=0.000001,penalty = 'l2', class_weight='balanced')

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs = -1)

scaler = StandardScaler()
X_std = scaler.fit_transform(X_training)

random_search.fit(X_std, y_training)
preds = random_search.predict_proba(scaler.transform(X_letter))
avg_pred_cab_member = preds.mean(axis = 0)
[print([random_search.classes_[x],avg_pred_cab_member[x]]) for x in np.argsort(avg_pred_cab_member)]

high_prob_cab_member = (preds>.1).sum(axis=0)
[print([random_search.classes_[x],high_prob_cab_member[x]]) for x in np.argsort(high_prob_cab_member)]
