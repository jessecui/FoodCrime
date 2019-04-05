from lxml import html
import requests


#Las Vegas
zipcodes = [89101, 89102, 89106, 89107, 89108, 89110, 89117, 89128, 89129, 89130, 89131, 89134, 89138, 89143, 89144, 89145, 89146, 89149, 89166]

for zipcode in zipcodes:
	zipcode = str(zipcode) 
	page = requests.get("https://www.bestplaces.net/crime/zip-code/nevada/las_vegas/"+zipcode)

	tree = html.fromstring(page.content)
	treeXPath = tree.xpath('//h5/text()')

	violentCrime = treeXPath[0][-5:-1]
	propertyCrime = treeXPath[1][-5:-1]

	print(zipcode + ": " + violentCrime + ": " + propertyCrime)