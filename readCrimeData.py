import csv, uszipcode
from uszipcode import Zipcode, SearchEngine
import pandas as pd
#os.chdir("~/UPenn18-19/CIS520/Project/")

#
#population = {}
#csvFile = open('./Population.csv', 'r')
#csvReader = csv.reader(csvFile)
#for row in csvReader:
#	break
#for row in csvReader:
#	population[row[0]] = int(row[1])
 

search = SearchEngine(simple_zipcode=True)

#Las Vegas
crime = {}
populationZipcode = {}
csvFile = open('LasVegas.csv', 'r')
csvReader = csv.reader(csvFile, delimiter=',')
for row in csvReader:
	break
for row in csvReader:
	try:
		coordinates = row[9]
		coordinates = coordinates[1:-2].split(", ")
		latitude = float(coordinates[0])
		longitude = float(coordinates[1])
		result = search.by_coordinates(latitude, longitude, returns = 1)[0]
		zipcode = result.zipcode
		population = result.population
		print(population)
		if zipcode in crime:
			crime[zipcode] += 1
		else:
			crime[zipcode] = 1
			populationZipcode[zipcode] = population
	except:
		pass
crimeExport = {'zipcode': [], 'crimeCount': [], 'population': []}
for zipcode in crime:
	crimeExport['zipcode'].append(zipcode)
	crimeExport['crimeCount'].append(crime[zipcode])
	crimeExport['population'].append(populationZipcode[zipcode])

crimeExportMatrix = pd.DataFrame(data=crimeExport)
crimeExportMatrix.to_pickle("LasVegasLabels")






