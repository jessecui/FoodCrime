# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import ast

# Load the dataframe    
df = pd.read_json('yelp_dataset/business.json', lines=True)
df_short = df.head()


# Metrics
# Is Open
# Review Count
# Stars
# One Hot Encoding on Attributes
# One Hot Encoding on Categories
# Hours per weekday for open and close (14 different numbers)
# Indicator variables indicating whether store is open on weekday

# Naive Bayes or Hidden Markov Model on the Data


# Shrink Data into core dataframe (we will keep business ID as a label)
df_core = df.drop(['address', 'city', 'latitude', 'longitude', 'state'], axis = 1)
df_core.columns

# Drop all rows where the zipcodes are not part of the label space (later when we get zip codes)

# One hot encoding on categories

rows, cols = df_core.shape
rows = 192609

print('PROCESSING CATEGORIES')
# Retrieve the categories
all_categories = []
for i in range(rows):
    if i % 10000 == 0:
        print(i)
    cat_string = df_core.loc[i, 'categories']
    
    try:
        categories = [x.strip() for x in cat_string.split(',')]
    except:
        pass    
    
    for category in categories:
        if category not in all_categories:
            all_categories.append(category)
    
all_categories.sort()

# Create and impute 1 on all restaurants where the category exists
categories_array = np.zeros((rows, len(all_categories)))


for i in range(rows):
    if i % 10000 == 0:
        print(i)
    keys_dict = dict((el,0) for el in all_categories)
    cat_string = df_core.loc[i, 'categories']
    categories = []
    
    try:
        categories = [x.strip() for x in cat_string.split(',')]
    except:
        pass
    
    for category in categories:
        keys_dict[category] = 1
        
    categories_array[i, :] = list(keys_dict.values())
    
categories_df = pd.DataFrame(categories_array, columns = all_categories)    
        

print('PROCESSING HOURS')
# Preprocess the hours column 
    
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']  
weekdays_avail = ['Monday_avail', 'Tuesday_avail', 'Wednesday_avail', 'Thursday_avail', 'Friday_avail', 'Saturday_avail', 'Sunday_avail']  
weekdays_dual = ['Monday_open', 'Monday_close',
                'Tuesday_open', 'Tuesday_close',
                'Wednesday_open', 'Wednesday_close',
                'Thursday_open', 'Thursday_close',
                'Friday_open', 'Friday_close',
                'Saturday_open', 'Saturday_close',
                'Sunday_open', 'Sunday_close']

total_hours_array = np.zeros((rows, 21))
  
for i in range(rows):
    if i % 10000 == 0:
        print(i)
    
    weekdays_avail_dict = dict((el,0) for el in weekdays)
    weekdays_hours_dict = dict((el,np.nan) for el in weekdays_dual)
    
    hours_dict = df_core.loc[i, 'hours']
    for weekday in weekdays:
        try:
            hours_string = hours_dict[weekday]        
            hours = [x.strip() for x in hours_string.replace(':', '-').split('-')]
            open_time = int(hours[0]) + (float(hours[1])/60)
            close_time = int(hours[2]) + (float(hours[3])/60)
            
            # Convert midnight times starting with 0 to starting at 24
            if close_time < 1:
                close_time = close_time + 24 
                
            weekdays_hours_dict[weekday + '_open'] = open_time
            weekdays_hours_dict[weekday + '_close'] = close_time
            
            # Mark the availability dataframe as available
            weekdays_avail_dict[weekday] = 1            
        except:
            pass
    hours_row = list(weekdays_hours_dict.values()) + list(weekdays_avail_dict.values())
    total_hours_array[i, :] = hours_row
total_hours_df = pd.DataFrame(total_hours_array, columns = weekdays_dual + weekdays_avail)

# Fill in the hours dataframe nan cells with the mean of the columns
total_hours_df = total_hours_df.fillna(total_hours_df.mean())
        
# Process the attributes column    
# Construct a list of all possible attributes
print("PROCESSING ATTRIBUTES COLUMN")
all_attributes = []
for i in range(rows):
    if i % 10000 == 0:
        print(i)
    att_dict = df_core.loc[i, 'attributes']
    if att_dict is not None:
        for key, value in att_dict.items():
            try:
                core_value = ast.literal_eval(value)
            except:
                core_value = value
            if type(core_value) == dict:
                for subkey in core_value.keys():
                    if key + '_' + subkey not in all_attributes:
                        all_attributes.append(key + '_' + subkey)
            else:
                if key not in all_attributes:
                    all_attributes.append(key)
    
all_attributes.sort()


# Create and impute 1 on all restaurants where the category exists
attributes_array = np.chararray((rows, len(all_attributes)), unicode = True, itemsize = 20)

# Fill in the attributes dataframe with the data
for i in range(rows):
    if i % 10000 == 0:
        print(i)
    att_df_row_dict = dict((el,'NA') for el in all_attributes)
    att_dict = df_core.loc[i, 'attributes']
    
    if att_dict is not None:
        for key, value in att_dict.items():
            try:
                core_value = ast.literal_eval(value)
            except:
                core_value = value
            if type(core_value) == dict:
                for subkey, subvalue in core_value.items():
                    att_df_row_dict[key + '_' + subkey] = str(subvalue)
            else:
                att_df_row_dict[key] = str(value)            
    attributes_array[i, :] = list(att_df_row_dict.values())

# Convert NP Array to dataframe
attributes_df = pd.DataFrame(attributes_array, columns = all_attributes)    

# Get one hot encoded features for the attributes table
print('ONE HOT ENCODING ATTRIBUTES')
attributes_df_encoded = pd.get_dummies(attributes_df).astype('int32')

# Merge all the tables together to one data set
print('CREATING FINAL DATASET')
df_super_core = df_core.drop(columns=['categories', 'hours', 'attributes'])
data_final = pd.concat([df_super_core, total_hours_df, categories_df, attributes_df_encoded], axis = 1)

names_df = data_final[['business_id', 'name']]
numerical_df = data_final.drop(columns=['name'])

data_final.to_csv('data_finalized.csv', index=False)
                
        
        

    




