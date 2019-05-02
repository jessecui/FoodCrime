# FoodCrime

FoodCrime is a student machine learning project that uses Yelp's dataset of restaurants and stores to predict crime. We used Yelp's dataset as our feature space and we queried property and violent crime rates per zipcode from bestplaces.net as our label space.

## Algorithm Structure

We used a ensemble approach that combines 3 types of models into a regressor for crime rates, namely one each for numerical regression, natural langauge processing, and computer vision. The three individual models are an XGBoost model for numerical/categorical business data (store attributes), a random forest regressor for text data (bag-of-words) in the form of reviews, and a CNN for image data. We split our data into a 60%, 30% and 10% split for training data for our models, training data for our ensemble model, and testing data.

## Directory Structure
We have 5 main folders, three directories for preprocessing and creating the features for each of the three data types (numerical-XGBoost:Numerical Data , text- RandomForest:Text Data, and image- CNN:Image Data), one for Labels, and one for Results. 

## Results
Our models show improvements from a baseline MSE involving random data. For the numerical model that uses XGBoost, we have a testing MSE of 294.36 and 349.52 for property and violent crime predictions respectively. For the random forest text model, we have a test MSE of 295.83, and for the image data, we have a test MSE of anywhere between the range of 268-398 depending on the type of images used.

For our final ensemble model, we retrieved the weights of the individual models as shown below. The text and numerical data models had the highest weights, largely attributed to the sparsity of image data for restaurants.

| Model  | Data | Weight |
| ------------- | ------------- | ------------- |
| Random Forest  | Text Review Data  | 0.9262 |
| XGBoost  | Business Attributes  | 0.7078 |
| CNN  | Food Images  | -0.1232 |
| CNN  | Drinks Images  | 0.0701 |
| CNN  | Outside Photos  | 0.0131 |
| CNN  | Inside Photos  | 0.0672 |