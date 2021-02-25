#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global Parameters
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1', error_bad_lines=False)
    dataset.columns = cols
    return dataset


def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset


def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove "See more"
    tweet = tweet.replace('See more', "")
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]

    return " ".join(lemma_words)


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"


# Load dataset
dataset = load_dataset("/Users/hamzahmahmood/PycharmProjects/OxHack/Sentiment Analysis/training data/training.csv",
                       ['target', 't_id', 'created_at', 'query', 'user', 'text'])
# Remove unwanted columns from dataset
n_dataset = remove_unwanted_cols(dataset, ['t_id', 'created_at', 'query', 'user'])
# Preprocess data
dataset.text = dataset['text'].apply(preprocess_tweet_text)
# Split dataset into Train, Test

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Training Naive Bayes model
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_predict_nb = NB_model.predict(X_test)
print(accuracy_score(y_test, y_predict_nb))

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)
print(accuracy_score(y_test, y_predict_lr))


# ------------------------------------------------#
# Applying model to Oxfess datatest
with open("/Users/hamzahmahmood/PycharmProjects/OxHack/Sentiment Analysis/test data/confessions.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', skipinitialspace=True)
    data = []
    for row in csv_reader:
        data.append(row)
    test_ds = pd.DataFrame.from_records(data, columns=["initials", "college", "dateofpost", "hashtag", "content",
                                                       "nameofpage", "1", "2", "3", "4", "5"])

test_ds = remove_unwanted_cols(test_ds, ["initials", "college", "1", "2", "3", "4", "5"])
test_ds['dateofpost'] = pd.to_datetime(test_ds['dateofpost'], format="%Y/%m/%d")
print(test_ds['dateofpost'])

# Creating text feature
test_ds.content = test_ds["content"].apply(preprocess_tweet_text).str.lower()

test_feature = tf_vector.transform(np.array(test_ds.iloc[:, 1]).ravel())

# Using Logistic Regression model for prediction
test_prediction_lr = LR_model.predict(test_feature)

test_result_ds = pd.DataFrame({'hashtag': test_ds.hashtag, 'prediction': test_prediction_lr,
                               "dateofpost": test_ds.dateofpost})
print(test_result_ds.groupby('prediction').count())
test_result = test_result_ds.groupby('dateofpost')
test_result.columns = ['hashtag', 'prediction', 'dateofpost', 'sum']
test_result.predictions = test_result['prediction']
#test_result.sum = test_result.sum('prediction')

print(test_result.head())

# Sum the prediction scores for each group
test_result_summed = test_result.sum('prediction')

print(test_result_summed.sort_values('dateofpost'))

sns.set_theme(style='darkgrid')
plt.xlabel("Time")
plt.ylabel("Mood")
plt.plot(test_result_summed.sort_values('dateofpost'))
plt.show()
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About

