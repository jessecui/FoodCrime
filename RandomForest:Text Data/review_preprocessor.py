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


