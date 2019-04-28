#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 01:11:13 2019

@author: jessecui
"""

import pandas as pd
import numpy as np
import pickle

df_full = pd.read_csv('data_finalized_with_city.csv')

# Drop the canadian ones
df_full['postal_code'] = pd.to_numeric(df_full['postal_code'], errors='coerce')
df_full = df_full[np.isfinite(df_full['postal_code'])]

df_crimes = pd.read_csv('top_zips_with_crimes.csv')
df_crimes = df_crimes.drop(['Unnamed: 0', 'Unnamed: 0.1', 'count'], axis=1)

df_full_with_crimes = df_full.merge(df_crimes, left_on='postal_code', right_on='zipcodes')
df_full_with_crimes = df_full_with_crimes.drop(['zipcodes'], axis=1)
df_full_with_crimes.to_csv('df_top_full_with_crimes.csv')



business_ids = df_full_with_crimes['business_id'].values
import random
random.Random(0).shuffle(business_ids)

business_ids_set = list(set(business_ids))

tenth_len = int(len(business_ids_set) / 10)
train_num = tenth_len * 6;
model_test_num = tenth_len * 9;

train_business_ids = business_ids_set[:train_num]
valid_business_ids = business_ids_set[train_num:model_test_num]
test_business_ids = business_ids_set[model_test_num:]

train_business_ids.sort()
valid_business_ids.sort()
test_business_ids.sort()

with open('train_valid_test_business_ids.pk', 'wb') as fp:
    pickle.dump((train_business_ids, valid_business_ids, test_business_ids), fp)
    
# ------------------------------------------------------------------------------    
# Now create the training and testing sets for the main regressor
train_business_ids, valid_business_ids, test_business_ids = pickle.load(open('train_valid_test_business_ids.pk', 'rb'))
df_full = pd.read_csv('df_top_full_with_crimes.csv')
df_full_core = df_full.drop(['postal_code', 'Unnamed: 0', 'Unnamed: 0.1', 'city', 'name', 'state'], axis=1)

# Subset into training and testing data
train_df = df_full_core[df_full_core['business_id'].isin(train_business_ids)].sort_values(by=['business_id']).drop_duplicates(subset='business_id').drop(['business_id'], axis=1)
valid_df = df_full_core[df_full_core['business_id'].isin(valid_business_ids)].sort_values(by=['business_id']).drop_duplicates(subset='business_id').drop(['business_id'], axis=1)
test_df = df_full_core[df_full_core['business_id'].isin(test_business_ids)].sort_values(by=['business_id']).drop_duplicates(subset='business_id').drop(['business_id'], axis=1)


# Subset into X and y
train_df_X = train_df[train_df.columns[:-2]]
train_df_y = train_df[train_df.columns[-2:]]
test_df_X = test_df[test_df.columns[:-2]]
test_df_y = test_df[test_df.columns[-2:]]
valid_df_X = valid_df[valid_df.columns[:-2]]
valid_df_y = valid_df[valid_df.columns[-2:]]

# Convert to numpy array tuple and save pickle
data_tuple = (train_df_X.values, train_df_y.values, 
              valid_df_X.values, valid_df_y.values, 
              test_df_X.values, test_df_y.values, list(train_df_X), list(train_df_y))
with open('main_data_tuple_2.pk', 'wb') as fp:
    pickle.dump(data_tuple, fp)