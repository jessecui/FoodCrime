# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt

df_full = pd.read_csv('data_finalized.csv')

df = df_full

names_df = df[['business_id', 'name']]
numerical_df = df.drop(columns=['name'])

m, d = df.shape

crimes_df = pd.DataFrame(np.array([[89108, 62.6, 64.8], 
                                 [89110, 56.6, 58.1],
                                 [89129, 41.1, 44.2],
                                 [89102, 71.9, 73.0],
                                 [89149, 38.2, 41.3]]), columns=['zip_code', 'violent_crime', 'property_crime'])

# Drop all non-us zipcodes
numerical_df['postal_code'] = pd.to_numeric(numerical_df['postal_code'], errors='coerce')
numerical_df = numerical_df[np.isfinite(numerical_df['postal_code'])]

# Only use restaurants in the zip code range
zip_codes = list(crimes_df['zip_code'].values)
numerical_df = numerical_df[numerical_df['postal_code'].isin(zip_codes)]

X_df = numerical_df.drop(columns=['business_id', 'postal_code'])
y_df = pd.merge(numerical_df[['business_id', 'postal_code']], crimes_df, how = 'left', left_on='postal_code', right_on='zip_code', sort=False)
y_df = y_df.iloc[:, 3:]

X = X_df.values
y = y_df.values

# PREDICTIONS FOR VIOLENT CRIMES
from sklearn.cross_validation import train_test_split

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
df_cols[1157] 

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