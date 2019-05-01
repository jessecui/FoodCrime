# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt
import re
from collections import Counter
import pickle

#Read in your restaurant file
df_reader = pd.read_csv('data_finalized.csv', chunksize = 40000)

zipcodes = Counter()

for df in df_reader:
	#Count number in each zipcode
	for index, row in df.iterrows():
		zipcode = row["postal_code"]
		if len(re.findall(r'^[\d{5}]', zipcode)) != 0:
			zipcodes[zipcode] += 1

	#Most common zipcodes
	n = 1000
	finalZipcodes = []
	for element in zipcodes.most_common(n):
		zipcode = element[0]
		finalZipcodes.append(zipcode)

print(finalZipcodes)

#Write zipcodes
finalZipcodes = sorted(finalZipcodes)
with open("zipcodes", "wb") as fp:
	pickle.dump(finalZipcodes, fp)