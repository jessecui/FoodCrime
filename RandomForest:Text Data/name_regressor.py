# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt

df_full = pd.read_csv('data_finalized.csv')

df = df_full.truncate(before = 0, after = 10000)

names_df = df[['business_id', 'name']]
numerical_df = df.drop(columns=['name'])

m, d = df.shape

crimes_df = pd.DataFrame(np.random.randint(0, 10, size=(m, 1)), columns=['crime_num'])

# TRAIN RANDOM FOREST REGRESSION MODEL ON RESTAURANT NAMES
# Note that this entire model was scrapped because the data was extremely sparse.
# Instead, we use reviews data to predict crimes which has much more crime data.

# First clean the text data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, m):
    if i % 100 == 0:
        print(i) 
    text = re.sub('[^a-zA-Z]', ' ', names_df['name'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)
    
# Create the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
y = crimes_df['crime_num'].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting random forest regression to the text data
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 8, max_depth = 5, random_state = 0)
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)



