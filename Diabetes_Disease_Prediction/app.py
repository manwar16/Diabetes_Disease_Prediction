# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:45:11 2020

@author: Mono
"""

# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the  Naive Bayes model model
filename = 'diabetesPredictionModel.pkl'
nb_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])
        
        data = np.array([[glucose, bp, st, bmi, age]])
        my_prediction = nb_model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)