# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt
import re
from collections import Counter
import pickle

df_full = pd.read_csv('data_finalized.csv', nrows=1000)

df = df_full

print(df_full)

zipcodes = Counter()

for index, row in df.iterrows():
	zipcode = row["zipcode"]
	zipcodes[zipcode] += 1

n = 10
finalZipcodes = []
for element in zipcodes.most_common(n):
	zipcode = element[0]
	finalZipcodes.append(zipcode)

finalZipcodes = sorted(finalZipcodes)
with open("zipcodes", "wb") as fp:
	pickle.dump(finalZipcodes, fp)










