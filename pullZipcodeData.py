from lxml import html
import requests
import pandas as pd

import re

#Las Vegas
with open("zipcodes", "rb") as fp:
	zipcodes = pickle.load(fp)

violentCrimes = []
propertyCrimes = []

for zipcode in zipcodes:
	zipcode = str(zipcode) 
	page = requests.get("https://www.bestplaces.net/crime/zip-code/nevada/las_vegas/"+zipcode)

	tree = html.fromstring(page.content)
	treeXPath = tree.xpath('//h5/text()')

	violentCrime = re.findall(r'\d{1,2}.\d', treeXPath[0])[-1]
	propertyCrime = re.findall(r'\d{1,2}.\d', treeXPath[1])[-1]

	violentCrimes.append(violentCrime)
	propertyCrimes.append(propertyCrime)

	print(zipcode + ": " + violentCrime + ": " + propertyCrime)

data = {"zipcodes": zipcodes, "violentCrimes": violentCrimes, "propertyCrimes": propertyCrimes}
df = pd.DataFrame(data=data)
df.to_pickle("zipcodeCrime")