from lxml import html
import requests
import pandas as pd
import pickle
import re
from uszipcode import Zipcode, SearchEngine

search = SearchEngine(simple_zipcode=True)

#Las Vegas
with open("zipcodes", "rb") as fp:
	zipcodes = pickle.load(fp)

stateAbbreviationDict = {}
with open("states.csv", "r") as fp:
	for line in fp.readlines():
		statesLine = line.split(",")
		stateName = statesLine[0].strip().replace("\"", "")
		stateAbbreviation = statesLine[1].strip().replace("\"", "")
		print(stateName)
		print(stateAbbreviation)
		stateAbbreviationDict[stateAbbreviation] = stateName

print(zipcodes)
violentCrimes = []
propertyCrimes = []

zipcodeCrime = []

for zipcode in zipcodes:
	try:
		zipcode = str(zipcode) 
		zipcodeDict = search.by_zipcode(zipcode).to_dict()
		stateAbbreviation = zipcodeDict["state"]
		stateName = stateAbbreviationDict[stateAbbreviation].replace(" ", "_")
		cities = zipcodeDict["common_city_list"]
		for city in cities:
			city = city.replace(" ", "_")

			page = requests.get("https://www.bestplaces.net/crime/zip-code/" + stateName + "/" + city + "/" + zipcode)
			print("https://www.bestplaces.net/crime/zip-code/" + stateName + "/" + city + "/" + zipcode)
			tree = html.fromstring(page.content)
			treeXPath = tree.xpath('//h5/text()')

			if len(re.findall(r'\d{1,2}.\d', treeXPath[1])) != 0:
				violentCrime = re.findall(r'\d{1,2}.\d', treeXPath[1])[-1]
				propertyCrime = re.findall(r'\d{1,2}.\d', treeXPath[2])[-1]
				zipcodeCrime.append(zipcode)
				violentCrimes.append(violentCrime)
				propertyCrimes.append(propertyCrime)
				print(zipcode + ": " + violentCrime + ": " + propertyCrime)
				break
	except:
		pass

data = {"zipcodes": zipcodeCrime, "violentCrimes": violentCrimes, "propertyCrimes": propertyCrimes}
df = pd.DataFrame(data=data)
df.to_pickle("zipcodeCrime")