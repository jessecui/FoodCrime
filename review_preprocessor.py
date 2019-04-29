# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import ast
import sys

# Get a list of business ID to look for
df_full = pd.read_csv('df_top_full_with_crimes.csv')

crimes_df = pd.read_csv('top_zips_with_crimes.csv')
crimes_df = crimes_df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'count'], axis=1)

zip_codes = list(crimes_df['zipcodes'].values)

# Retrive the business IDs and drop the canadian ones
df_bis = df_full[['business_id', 'postal_code']]
df_bis['postal_code'] = pd.to_numeric(df_bis['postal_code'], errors='coerce')
df_bis = df_bis[np.isfinite(df_bis['postal_code'])]

df_bis = df_bis[df_bis['postal_code'].isin(zip_codes)]
business_ids = list(df_bis.business_id.unique())

# Load the dataframe    
max_records = 1e5
df = pd.read_json('review.json', lines=True, chunksize=max_records)

filtered_data = pd.DataFrame()
count = 0

for df_chunk in df:
    try:
        sub_chunk = df_chunk[['business_id', 'text']]
        sub_chunk = sub_chunk[sub_chunk['business_id'].isin(business_ids)]
        count += sub_chunk.shape[0]
        print(count)       
        filtered_data = pd.concat([filtered_data, sub_chunk])
    except ValueError:
        print('Some messages cannot be parsed')
   
# Save the data
#filtered_data.to_pickle('filtered_reviews_data.pk')        

# Combine the reviews that have the same business ID and concatenate their reviews
review_df = filtered_data        
review_df = review_df.applymap(str)
shortened_reviews = review_df.groupby('business_id')['text'].apply(lambda x: "%s" % ' '.join(x))
review_df = shortened_reviews.to_frame()
review_df.reset_index(level=0, inplace=True)

df = df_full

numerical_df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'city', 'name', 'state'], axis=1)

m, d = df.shape

# Drop all non-us zipcodes
numerical_df['postal_code'] = pd.to_numeric(numerical_df['postal_code'], errors='coerce')
numerical_df = numerical_df[np.isfinite(numerical_df['postal_code'])]

# Only use restaurants in the zip code range
numerical_df = numerical_df[numerical_df['postal_code'].isin(zip_codes)].drop_duplicates(subset='business_id')
bid_crimes = pd.merge(numerical_df[['business_id', 'postal_code']], crimes_df, how = 'left', left_on='postal_code', right_on='zipcodes', sort=False).drop_duplicates(subset='business_id')

# merge the Business ID v crimes data to the reviews data
total_df = pd.merge(review_df, bid_crimes, how = 'left', on='business_id')
X_df = total_df.drop(['business_id', 'postal_code', 'zipcodes', 'violentCrimes', 'propertyCrimes'], axis = 1)
y_df = total_df.drop(['business_id', 'text', 'postal_code', 'zipcodes'], axis  = 1)
# Word 2 vec

# TRAIN RANDOM FOREST REGRESSION MODEL ON REVIEW TEXDT
# Later use RNNs

# First clean the text data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer




# Convert DF to numpy char array

list_text = list(X_df['text'].values)

# Trunacte all Reviews to 5000 characters
list_text = [x[:5000] for x in list_text]

# Only pick 5000 random reviews for corpus (to save time)
# Run on all in the future
corpus = []
list_text_length = len(list_text)
for i in range(0, list_text_length):
    if i % 1 == 0:
        print(i) 
    text = re.sub('[^a-zA-Z]', ' ', list_text[i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)
    
    
# Save the corpus so we don't have to rerun it again    
import pickle
with open('corpus_full_1.pk', 'wb') as fp:
    pickle.dump(corpus, fp)        
    
# Create the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = y_df.values

# Save the count vectorizer
with open('count_vectorizer_2.pk', 'wb') as fp:
    pickle.dump(cv, fp)  
    
cv = pickle.load(open('count_vectorizer_2.pk', 'rb'))    
corpus = pickle.load(open('corpus_full_1.pk', 'rb'))

# Join the resulting vectorized text data to the business ids
transformed_array = np.hstack((X, y))    
transformed_df = pd.DataFrame(data = transformed_array)
business_ids_col = total_df['business_id']

id_transformed_df = pd.concat([business_ids_col, transformed_df], axis=1)
id_transformed_df.to_csv('id_transformed_df.csv')


# ---------------------------------------------------------------------------------
id_transformed_df = pd.read_csv('id_transformed_df.csv')
id_transformed_df = id_transformed_df.drop(['Unnamed: 0'], axis=1)

# Load the train, validation and testing business ids
train_business_ids, valid_business_ids, test_business_ids = pickle.load(open('train_valid_test_business_ids.pk', 'rb'))
test_business_ids_super = valid_business_ids + test_business_ids

train_df_reviews = id_transformed_df[id_transformed_df['business_id'].isin(train_business_ids)].sort_values(by=['business_id']).drop_duplicates(subset='business_id').drop(['business_id'], axis=1)
validate_df_reviews = id_transformed_df[id_transformed_df['business_id'].isin(valid_business_ids)].sort_values(by=['business_id']).drop_duplicates(subset='business_id')
test_df_reviews = id_transformed_df[id_transformed_df['business_id'].isin(test_business_ids)].sort_values(by=['business_id']).drop_duplicates(subset='business_id').drop(['business_id'], axis=1)
test_df_reviews_wid = id_transformed_df[id_transformed_df['business_id'].isin(test_business_ids)].sort_values(by=['business_id']).drop_duplicates(subset='business_id')

# Join the validate_df_reviews to get the missing business_ids and fill those with random values
valid_bi_df = pd.DataFrame(data = valid_business_ids, columns=['business_id'])
valid_df_reviews_full = valid_bi_df.merge(validate_df_reviews, on='business_id', how='left')

#Impute missing values with mean
valid_df_reviews_full = valid_df_reviews_full.fillna(valid_df_reviews_full.mean())

test_super_df_reviews = id_transformed_df[id_transformed_df['business_id'].isin(test_business_ids_super)].sort_values(by=['business_id']).drop_duplicates(subset='business_id').drop(['business_id'], axis=1)

# Splitting the dataset into the Training set and Test set
X_train = train_df_reviews[train_df_reviews.columns[:-2]].values
y_train = train_df_reviews[train_df_reviews.columns[-2:]].values

X_test = test_super_df_reviews[test_super_df_reviews.columns[:-2]].values
y_test = test_super_df_reviews[test_super_df_reviews.columns[-2:]].values

X_valid = valid_df_reviews_full[valid_df_reviews_full.columns[:-2]].drop(['business_id'], axis=1).values
y_valid = valid_df_reviews_full[valid_df_reviews_full.columns[-2:]].values

X_test_basic = test_df_reviews[test_df_reviews.columns[:-2]].values
y_test_basic = test_df_reviews[test_df_reviews.columns[-2:]].values

# Fitting random forest regression to the text data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Perform crossfold validation
estimators = [3, 5, 10, 20, 50, 100, 200]
max_depth = [None, 3, 5, 10, 20, 50]
results = {}

kf = KFold(n_splits = 5, random_state = 0, shuffle = True)
for estimator_num in estimators:
    print('Parameter', estimator_num)
    sum_score = 0
    for train_index, test_index in kf.split(X_train):
        cfv_trainX, cfv_validX = X_train[train_index], X_train[test_index]
        cfv_trainy, cfv_validy = y_train[train_index], y_train[test_index]
        
        regressor = RandomForestRegressor(n_estimators = estimator_num, max_depth = 10, random_state = 0, verbose = True)
        regressor.fit(cfv_trainX, cfv_trainy)
        predictions = regressor.predict(cfv_validX)
        
        mse = mean_squared_error(predictions, cfv_validy)
        print('MSE ', mse)
            
        sum_score += mse
        
        break
        
    final_score = sum_score
    results[(estimator_num)] = final_score

# Choose the best 
best_estimators, best_depth = 50, 10
best_model = RandomForestRegressor(n_estimators = best_estimators, max_depth = best_depth, random_state = 0, verbose=True, n_jobs = 6)
best_model.fit(X_train, y_train)

best_model2 = RandomForestRegressor(n_estimators = 10, max_depth = 10, random_state = 0, verbose=True, n_jobs = 6)
best_model2.fit(X_train, y_train)

# Pickle the model
# Predict the validation scores
validation_predictions_rf = best_model.predict(X_valid)
test_predictions_rf = best_model.predict(X_test_basic)
with open('predictions_text_rf.pk', 'wb') as fp:
    pickle.dump((validation_predictions_rf, test_predictions_rf), fp)  
    
# Prediction
from sklearn.metrics import mean_squared_error
y_pred = best_model2.predict(X_test)
print("MSE RF Test: ", mean_squared_error(y_test, y_pred))

y_pred_train = best_model.predict(X_train)
print("MSE RF Train: ", mean_squared_error(y_train, y_pred_train))

print("MSE: ", mean_squared_error(y_test_basic[:, 0], validation_predictions_rf[:, 0]))

with open('best_reviews_regressor.pk', 'wb') as fp:
    pickle.dump(best_model, fp)  

# Prediction with dummy data
X_train_random = np.random.uniform(-1, 1, (X_train.shape))
X_test_random = np.random.uniform(-1, 1, (X_test.shape))
dumb_regressor = RandomForestRegressor(n_estimators = best_estimators, max_depth = best_depth, random_state = 0, verbose=True, n_jobs = 6)
dumb_regressor.fit(X_train_random, y_train)

y_pred_dumb = dumb_regressor.predict(X_test_random)
print("BASELINE MSE RF: ", mean_squared_error(y_test, y_pred_dumb))

# Interpretations
importances = best_model2.feature_importances_
important_indices = sorted(range(len(importances)), key=lambda i: importances[i])[-100:]

words = cv.get_feature_names()
top_words = [words[i] for i in important_indices]

# Use SHAP to better interpret model
import shap
shap.initjs()

explainer = shap.TreeExplainer(best_model2)
X_train_df = pd.DataFrame(data = X_train, columns=words)
shap_values = explainer.shap_values(X_train_df)

average_shaps = np.median(shap_values, axis=0)
words_array = np.array(words)
array_directions = np.transpose(np.vstack((words_array, average_shaps)))

directions_df = pd.DataFrame(data = array_directions, columns=['Word', 'Value'])
directions_df["Value"] = pd.to_numeric(directions_df["Value"])*-1
directions_df = directions_df.reindex(directions_df.Value.abs().sort_values(ascending=False).index)
directions_df.to_csv('words_correlation.csv', index=False)

shap.summary_plot(shap_values, X_train_df, max_display = 50)


