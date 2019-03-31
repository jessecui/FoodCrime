# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_json('yelp_dataset/business.json', lines=True)
print('hi')

df_short = df.head()

df['postal_code'].nunique()
df['state'].unique()

df_counts = df.groupby('state')['business_id'].nunique()
df_counts = df.groupby(['state','postal_code'])['business_id'].nunique().to_frame()
df_counts = df_counts.sort_values(['business_id'], ascending = False)