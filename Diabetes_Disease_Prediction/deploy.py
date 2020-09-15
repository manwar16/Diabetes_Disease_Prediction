# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:52:59 2020

@author: Mono
"""

import pandas as pd
import numpy as np
import pickle
#from sklearn.impute import SimpleImputer


data_frame = pd.read_csv(r"C:\Users\IT_SHOP\deployment\diabetes.csv")

data_frame['glucose_conc'].fillna(data_frame['glucose_conc'].mean(), inplace=True)
data_frame['diastolic_bp'].fillna(data_frame['diastolic_bp'].mean(), inplace=True)
data_frame['skin_thickness'].fillna(data_frame['skin_thickness'].mean(), inplace=True)
data_frame['bmi'].fillna(data_frame['bmi'].mean(), inplace=True)



#for model building
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X = data_frame.drop(columns='diabetes')
y = data_frame['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)



# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())

# creating a pickle file 
filename = 'diabetesPredictionModel.pkl'
pickle.dump(nb_model, open(filename, 'wb'))


