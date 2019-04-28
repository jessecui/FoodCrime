# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt
import pickle

df_full = pd.read_csv('data_finalized.csv')

df_full = pd.read_csv('Pittsburg_data_full.csv')
df = df_full

names_df = df[['business_id', 'name']]
numerical_df = df.drop(columns=['name'])

m, d = df.shape

crimes_df = pd.DataFrame(np.array([[89108, 62.6, 64.8], 
                                 [89110, 56.6, 58.1],
                                 [89129, 41.1, 44.2],
                                 [89102, 71.9, 73.0],
                                 [89149, 38.2, 41.3]]), columns=['zipcodes', 'violent_crime', 'property_crime'])

crimes_df = pd.read_csv('top_zips_with_crimes.csv')

# Drop all non-us zipcodes
numerical_df['postal_code'] = pd.to_numeric(numerical_df['postal_code'], errors='coerce')
numerical_df = numerical_df[np.isfinite(numerical_df['postal_code'])]

# Only use restaurants in the zip code range
zip_codes = list(crimes_df['zipcodes'].values)
numerical_df = numerical_df[numerical_df['postal_code'].isin(zip_codes)]

X_df = numerical_df.drop(columns=['business_id', 'postal_code'])
y_df = pd.merge(numerical_df[['business_id', 'postal_code']], crimes_df, how = 'left', left_on='postal_code', right_on='zipcodes', sort=False)
y_df = y_df.iloc[:, 2:]

X = X_df.values
y = y_df.values

# PREDICTIONS FOR PROPERTY CRIMES
# ------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib as plt
import pickle
# Start training the data
X_train, y_train, X_test, y_test, X_test2, y_test2, X_cols, y_cols = pickle.load(open('main_data_tuple_2.pk', 'rb'))


X_test_super = np.vstack((X_test, X_test2))
y_test_super = np.vstack((y_test, y_test2))

y_train_property = y_train[:,0]
y_test_property = y_test_super[:,0]
y_train_violent = y_train[:,1]
y_test_violent = y_test_super[:,1]

# Feature scale the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 1:17] = sc_X.fit_transform(X_train[:, 1:17])
X_test[:, 1:17] = sc_X.transform(X_test[:, 1:17])
X_test_super[:, 1:17] = sc_X.transform(X_test_super[:, 1:17])


# Train full model with XGBoost
import xgboost as xgb

#dtrain = xgb.DMatrix(X_train, y_train_property, missing = -999)

# Training and cross-fold validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
etas = [0.1, 0.3, 0.5, 0.7, 0.9]
max_depth = [6]
lambdas = [0.3, 1, 3, 10]


results = {}
kf = KFold(n_splits = 3, random_state = 0, shuffle = True)
for eta_num in lambdas:
    print('Parameter: ', eta_num)
    sum_score = 0
    for train_index, test_index in kf.split(X_train):
        cfv_trainX, cfv_validX = X_train[train_index], X_train[test_index]
        cfv_trainy, cfv_validy = y_train_property[train_index], y_train_property[test_index]

        xgb_params = {'eta': 0.5, 'max_depth': 6, 'lambda': eta_num}
        dtrain = xgb.DMatrix(cfv_trainX, cfv_trainy, missing = -999)
        dvalid = xgb.DMatrix(cfv_validX)
        
        bst1 = xgb.train(params = xgb_params, dtrain=dtrain)
        y_valid_pred = bst1.predict(dvalid)
        
        mse = mean_squared_error(y_valid_pred, cfv_validy)
        print('MSE ', mse)
        
        sum_score += mse
        
    final_score = sum_score / 3
    results[eta_num] = final_score


# ------------------------------------------------------------
# ------------------------------------------------------------
from sklearn.metrics import mean_squared_error
# Picking the best model
dtrain = xgb.DMatrix(X_train, y_train_property, missing = -999)
best_model1 = xgb.train(params = {'eta': 0.5, 'max_depth': 6, 'lambda': 3}, 
                        dtrain = dtrain)

# Training error
dtrainX = xgb.DMatrix(X_train)
train_predictions = best_model1.predict(dtrainX)
mse = mean_squared_error(y_train_property, train_predictions)
print('Training Error', mse)

# Testing error
dtestX = xgb.DMatrix(X_test_super)
test_predictions = best_model1.predict(dtestX)
mse = mean_squared_error(y_test_property, test_predictions)
results1 = np.vstack((y_test_property, test_predictions))
print('Testing Error', mse)

# Baseline error
X_train_random = np.random.uniform(-1, 1, (X_train.shape))
X_test_random = np.random.uniform(-1, 1, (X_test_super.shape))

r_dtrain = xgb.DMatrix(X_train_random, y_train_property, missing = -999)
r_bst = xgb.train(params = {'eta': 0.5, 'max_depth': 6, 'lambda': 3}, dtrain = r_dtrain)
r_dtest = xgb.DMatrix(X_test_super)
r_y_pred = r_bst.predict(r_dtest)
print('MEAN SQUARED ERROR BASELINE Property (RANDOM): ', mean_squared_error(y_test_property, r_y_pred))

# Validation Predictions
dvalidX = xgb.DMatrix(X_test)
validation_predictions = best_model1.predict(dvalidX)
mean_squared_error(y_test[:, 1], validation_predictions)

with open('violent_predictions_main.pk', 'wb') as fp:
    pickle.dump(validation_predictions, fp)


dtest = xgb.DMatrix(X_test)
y_pred = bst.predict(dtest)

# Evaluating

print('MEAN SQUARED ERROR Property: ', mean_squared_error(y_test_property, y_pred))

# MSE with random inputs
X_train_random = np.random.uniform(-1, 1, (X_train.shape))
X_test_random = np.random.uniform(-1, 1, (X_test.shape))

r_dtrain = xgb.DMatrix(X_train_random, y_train_property, missing = -999)
r_bst = xgb.train(params = {}, dtrain = r_dtrain)
r_dtest = xgb.DMatrix(X_test_random)
r_y_pred = r_bst.predict(r_dtest)
print('MEAN SQUARED ERROR BASELINE Property (RANDOM): ', mean_squared_error(y_test_property, r_y_pred))


# Use SHAP to better interpret model
import shap
shap.initjs()

explainer = shap.TreeExplainer(best_model1)
X_train_df = pd.DataFrame(data = X_train, columns=X_cols)
shap_values = explainer.shap_values(X_train_df)

shap.summary_plot(shap_values, X_train_df, max_display = 50)


# PREDICTIONS FOR VIOLENT CRIMES
# Feature scale the data
# In the future avoid feature scaling the dummy variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 1:17] = sc_X.fit_transform(X_train[:, 1:17])
X_test[:, 1:17] = sc_X.transform(X_test[:, 1:17])


# Train full model with XGBoost

dtrain = xgb.DMatrix(X_train, y_train_violent, missing = -999)

# Training
bstv = xgb.train(params = {}, dtrain = dtrain)

# Testing
dtest = xgb.DMatrix(X_test)
y_pred = bstv.predict(dtest)

# Evaluating
from sklearn.metrics import mean_squared_error
print('MEAN SQUARED ERROR VIOLENT: ', mean_squared_error(y_test_violent, y_pred_scaled_v))

# MSE with random inputs
X_train_random = np.random.uniform(-1, 1, (X_train.shape))
X_test_random = np.random.uniform(-1, 1, (X_test.shape))

r_dtrain = xgb.DMatrix(X_train_random, y_train_violent, missing = -999)
r_bst = xgb.train(params = {}, dtrain = r_dtrain)
r_dtest = xgb.DMatrix(X_test_random)
r_y_pred = r_bst.predict(r_dtest)
print('MEAN SQUARED ERROR BASELINE PROPERTY (RANDOM): ', mean_squared_error(y_test_violent, r_y_pred_scaled))


# Interpreting model
xgb.plot_importance(bst, max_num_features = 20)
scores = bst.get_fscore()

df_cols = list(X_df.columns)
df_cols[1157] 

# Use SHAP to better interpret model
import shap
shap.initjs()

explainer = shap.TreeExplainer(bst)
X_train_df = pd.DataFrame(data = X_train, columns=X_cols)
shap_values = explainer.shap_values(X_train_df)

shap.summary_plot(shap_values, X_train_df, max_display = 50)

# Conglomerate zip codes
df_full = pd.read_csv('df_top_full_with_crimes.csv')
df_full_core = df_full.drop(['Unnamed: 0', 'Unnamed: 0.1', 'city', 'name', 'state'], axis=1)
zips = list(set(df_full['postal_code'].values))

avg_df = df_full_core.groupby(['postal_code']).mean()

avg_X_vals = avg_df.values[:, :-2]
avg_X_vals[:, 17:] = np.round(avg_X_vals[:, 17:])
avg_y_vals_v = avg_df.values[:, -1]

d_avg_X = xgb.DMatrix(avg_X_vals)
avg_predictions = best_model1.predict(d_avg_X)
mean_squared_error(avg_y_vals_v, avg_predictions)





combined = np.transpose(np.vstack((test_zips, y_pred_scaled)))
combined_df = pd.DataFrame(data = combined, columns=['zipcode', 'crime_prediction'])
combined_df_gb = combined_df.groupby(['zipcode']).mean()

zipcrimes_df = pd.read_csv('zipcodeCrimes.csv').drop(['Unnamed: 0', 'propertyCrimes'], axis=1)

true_crimes = combined_df_gb.merge(zipcrimes_df, how='left', left_on='zipcode', right_on='zipcodes')
mean_squared_error(true_crimes.crime_prediction.values, true_crimes.violentCrimes.values)