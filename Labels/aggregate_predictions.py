def aggregate_predictions(y_true, y_pred, zipcode_crime, violent_crimes):
	
	if violent_crimes == True:
		for zipcode in zipcode_crime['zipcode']:
			y_pred_subset = y_pred[(y_true['postal_code'] == zipcode)]
			y_pred_subset_mean = y_pred_subset.mean()
			print(y_pred_subset_mean, zipcode_crime[zipcode, 'violentCrimes'])
	else:
		pass
