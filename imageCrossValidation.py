import pandas as pd
import sklearn
import numpy as np

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

#X that have image and X that do not - please do this Faraz
X_image = X_df.iloc[10:, :]
X_no_image = X_df.iloc[0:10, :]

#Cosine Similarity Version
from sklearn.metrics.pairwise import cosine_similarity

#Get the descriptors of restaurants
binary_data_image = np.asmatrix(X_image.iloc[:, 24:])
binary_data_no_image = np.asmatrix(X_no_image.iloc[:, 24:])


#Normalize them
binary_data_image = np.divide(binary_data_image, (np.linalg.norm(binary_data_image, axis = 0) + 0.0001))
binary_data_no_image = np.divide(binary_data_no_image, (np.linalg.norm(binary_data_no_image, axis = 0) + 0.0001))

#Compute the similarity
similarity_matrix = cosine_similarity(binary_data_no_image, binary_data_image)
best_matches = np.argmax(similarity_matrix, axis = 0)
#best_matches is for each restaurant, which index of image restaurant is the best?