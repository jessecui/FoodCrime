# -*- coding: utf-8 -*-

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition

df = pd.read_csv('Charlotte_data_full.csv')

#for df in df_reader:
	#Count number in each zipcode
	#df_initial = df
	#break

#df_full = df_initial

#for df in df_reader:
	#df_full = pd.concat([df_full, df], axis = 0)

names_df = df[['business_id', 'name']]
numerical_df = df.drop(columns=['name', 'Unnamed: 0', 'Unnamed: 0.1'])

crimes_df = pd.read_csv("zipcodeCrimes.csv")

# Drop all non-us zipcodes
#numerical_df['postal_code'] = pd.to_numeric(numerical_df['postal_code'], errors='coerce')
#numerical_df = numerical_df[np.isfinite(numerical_df['postal_code'])]

# Only use restaurants in the zip code range
crimes_df = crimes_df.drop(columns=['Unnamed: 0'])
zip_codes = list(crimes_df['zipcodes'].values)
print(crimes_df.columns)
numerical_df = numerical_df[numerical_df['postal_code'].isin(zip_codes)]


X_df = numerical_df.drop(columns=['business_id', 'postal_code', 'city', 'state'])
y_df = pd.merge(numerical_df[['business_id', 'postal_code']], crimes_df, how = 'left', left_on='postal_code', right_on='zipcodes', sort=False)
print(y_df.columns)
y_df = y_df.iloc[:, 2:]

print(X_df.columns[0:24])
print(y_df.values)

X = X_df.values
y = y_df.values
m, d = X_df.shape

#COLLBORATIVE FILTERING
binary_data = np.asmatrix(X[:, 24:])
print(binary_data)

#feature_norm = np.linalg.norm(binary_data, axis = 0)
#binary_data = binary_data[:, (feature_norm != 0)]
u, s, vh = np.linalg.svd(a = binary_data, full_matrices = False)
binary_data_k = u * np.diag(s) * vh
print(binary_data_k)

k_max = len(s)
num_k = 5
k_error_violent = []

#Cross Validation on number of k
for k in [int(k_max / num_k) * (i + 1) for i in range(num_k)]:
	print(k)
	s_k = s[0:k]
	u_k = u[:, 0:k]
	vh_k = vh[0:k, :]
	binary_data_k = u_k * np.diag(s_k) * vh_k
	X[:, 24:] = binary_data_k

	#Cosine Similarity Version
	#from sklearn.metrics.pairwise import cosine_similarity
	#binary_data = np.divide(binary_data, np.linalg.norm(binary_data, axis = 0))
	#similarity_matrix = cosine_similarity(binary_data.transpose())
	#print(similarity_matrix)

	# PREDICTIONS FOR VIOLENT CRIMES
	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y[:, 0], test_size = 0.20, random_state = 1)

	# Feature scale the data
	# In the future avoid feature scaling the dummy variables
	from sklearn.preprocessing import StandardScaler
	sc_X = StandardScaler()
	X_train[:, 0:17] = sc_X.fit_transform(X_train[:, 0:17])
	X_test[:, 0:17] = sc_X.transform(X_test[:, 0:17])
	sc_y = StandardScaler()
	y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

	# Train full model with XGBoost
	import xgboost as xgb

	dtrain = xgb.DMatrix(X_train, y_train, missing = -999)

	# Training
	bst = xgb.train(params = {}, dtrain = dtrain)

	# Testing
	dtest = xgb.DMatrix(X_test)
	y_pred = bst.predict(dtest)
	y_pred_scaled = sc_y.inverse_transform(y_pred)

	# Evaluating
	from sklearn.metrics import mean_squared_error
	print('MEAN SQUARED ERROR VIOLENT: ', mean_squared_error(y_test, y_pred_scaled))
	aggregate_predictions(y_test, y_pred, crime_df, True)

	# MSE with random inputs
	X_train_random = np.random.normal(size = X_train.shape)
	X_test_random = np.random.normal(size = X_test.shape)

	r_dtrain = xgb.DMatrix(X_train_random, y_train, missing = -999)
	r_bst = xgb.train(params = {}, dtrain = r_dtrain)
	r_dtest = xgb.DMatrix(X_test_random)
	r_y_pred = r_bst.predict(r_dtest)
	r_y_pred_scaled = sc_y.inverse_transform(r_y_pred)
	print('MEAN SQUARED ERROR BASELINE VIOLENT (RANDOM): ', mean_squared_error(y_test, r_y_pred_scaled))

	k_error_violent.append(mean_squared_error(y_test, y_pred_scaled))

print(k_error_violent)
k_index = k_error_violent.index(min(k_error_violent))
k = int(k_max / num_k) * (k_index + 1)

s_k = s[0:k]
u_k = u[:, 0:k]
vh_k = vh[0:k, :]
binary_data_k = u_k * np.diag(s_k) * vh_k
X[:, 24:] = binary_data_k


#Cosine Similarity Version
#from sklearn.metrics.pairwise import cosine_similarity
#binary_data = np.divide(binary_data, np.linalg.norm(binary_data, axis = 0))
#similarity_matrix = cosine_similarity(binary_data.transpose())
#print(similarity_matrix)

# PREDICTIONS FOR VIOLENT CRIMES
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y[:, 0], test_size = 0.20, random_state = 1)

# Feature scale the data
# In the future avoid feature scaling the dummy variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 0:17] = sc_X.fit_transform(X_train[:, 0:17])
X_test[:, 0:17] = sc_X.transform(X_test[:, 0:17])
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))


# Train full model with XGBoost
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, y_train, missing = -999)

# Training
bst = xgb.train(params = {}, dtrain = dtrain)

# Testing
dtest = xgb.DMatrix(X_test)
y_pred = bst.predict(dtest)
y_pred_scaled = sc_y.inverse_transform(y_pred)

# Evaluating
from sklearn.metrics import mean_squared_error
print('MEAN SQUARED ERROR VIOLENT: ', mean_squared_error(y_test, y_pred_scaled))

# MSE with random inputs
X_train_random = np.random.uniform(-1, 1, (X_train.shape))
X_test_random = np.random.uniform(-1, 1, (X_test.shape))

r_dtrain = xgb.DMatrix(X_train_random, y_train, missing = -999)
r_bst = xgb.train(params = {}, dtrain = r_dtrain)
r_dtest = xgb.DMatrix(X_test_random)
r_y_pred = r_bst.predict(r_dtest)
r_y_pred_scaled = sc_y.inverse_transform(r_y_pred)
print('MEAN SQUARED ERROR BASELINE VIOLENT (RANDOM): ', mean_squared_error(y_test, r_y_pred_scaled))


# Interpreting model
xgb.plot_importance(bst, max_num_features = 20)
scores = bst.get_fscore()

df_cols = list(X_df.columns)

# Use SHAP to better interpret model
import shap
shap.initjs()

explainer = shap.TreeExplainer(bst)
X_train_df = pd.DataFrame(data = X_train, columns=X_df.columns)
shap_values = explainer.shap_values(X_train_df)

shap.summary_plot(shap_values, X_train_df, max_display = 50)


# PREDICTIONS FOR PROPERTY CRIMES
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y[:, 1], test_size = 0.20, random_state = 1)

# Feature scale the data
# In the future avoid feature scaling the dummy variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 0:17] = sc_X.fit_transform(X_train[:, 0:17])
X_test[:, 0:17] = sc_X.transform(X_test[:, 0:17])
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))


# Train full model with XGBoost

dtrain = xgb.DMatrix(X_train, y_train, missing = -999)

# Training
bst = xgb.train(params = {}, dtrain = dtrain)

# Testing
dtest = xgb.DMatrix(X_test)
y_pred = bst.predict(dtest)
y_pred_scaled = sc_y.inverse_transform(y_pred)

# Evaluating
from sklearn.metrics import mean_squared_error
print('MEAN SQUARED ERROR PROPERTY: ', mean_squared_error(y_test, y_pred_scaled))

# MSE with random inputs
X_train_random = np.random.uniform(-1, 1, (X_train.shape))
X_test_random = np.random.uniform(-1, 1, (X_test.shape))

r_dtrain = xgb.DMatrix(X_train_random, y_train, missing = -999)
r_bst = xgb.train(params = {}, dtrain = r_dtrain)
r_dtest = xgb.DMatrix(X_test_random)
r_y_pred = r_bst.predict(r_dtest)
r_y_pred_scaled = sc_y.inverse_transform(r_y_pred)
print('MEAN SQUARED ERROR BASELINE PROPERTY (RANDOM): ', mean_squared_error(y_test, r_y_pred_scaled))


# Interpreting model
xgb.plot_importance(bst, max_num_features = 20)
scores = bst.get_fscore()

df_cols = list(X_df.columns)
df_cols[1157] 

# Use SHAP to better interpret model
import shap
shap.initjs()

explainer = shap.TreeExplainer(bst)
X_train_df = pd.DataFrame(data = X_train, columns=X_df.columns)
shap_values = explainer.shap_values(X_train_df)

shap.summary_plot(shap_values, X_train_df, max_display = 50)