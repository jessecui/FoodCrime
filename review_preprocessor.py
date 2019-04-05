# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import ast

# Load the dataframe    
df = pd.read_json('yelp_dataset/review.json', lines=True)
df_short = df.head()