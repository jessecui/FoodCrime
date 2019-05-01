#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:21:28 2019

@author: jessecui
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('data_finalized_with_city.csv')

# Distribution of store ratings
sub_df = df[['stars', 'business_id']]

rating = df['stars'].values
example = sub_df.groupby(['stars']).agg('count')

plt.hist(rating, bins='auto')  # arguments are passed to np.histogram
plt.title("Ratings Histogram")
plt.show()

# Opening times
sub_df = df[['Monday_open', 'Tuesday_open', 'Wednesday_open', 'Thursday_open', 'Friday_open', 'Saturday_open', 'Sunday_open']]
open_array = sub_df.values

open_array = open_array.flatten().astype(int)
plt.hist(open_array)  # arguments are passed to np.histogram
plt.title("Store Opening Times (Military Hours)")
plt.show()

unique, counts = np.unique(open_array, return_counts=True)
dict(zip(unique, counts))

# Mean metrics
mean_df = df.iloc[:, :30]
mean_df = mean_df.drop(columns = ['Unnamed: 0', 'business_id', 'city', 'name', 'postal_code', 'state'])
mean_df = mean_df.mean()