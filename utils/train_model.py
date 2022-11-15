"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y = train_data[['load_shortfall_3h']]
X = train_data[['Bilbao_rain_1h', 'Seville_temp_max', 'Seville_temp', 'Barcelona_temp']]

# Fit model
lm_regression = LinearRegression()

lm_regression.fit(X, y)

# Pickle model for use within our API
save_path = '../assets/trained-models/flask_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
