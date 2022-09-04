from django.shortcuts import render

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def home(request):
    return render(request , 'home.html')

def predict(request):
    return render(request , 'predict.html')

def result(request):
    diabetes_dataset = pd.read_csv(r"static/diabetes.csv")
    #separating data and labels
    X = diabetes_dataset.drop(columns= 'Outcome' , axis=1)
    Y = diabetes_dataset['Outcome']

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = diabetes_dataset['Outcome']
    X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2 , stratify = Y, random_state = 2)
    classifier = svm.SVC(kernel = 'linear')
    #training the support vector machine classifier
    classifier.fit(X_train , Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    input_data =(val1 ,val2 ,val3 ,val4 ,val5 ,val6 ,val7 ,val8)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)

    res2=""

    if (prediction[0] ==0):
        res2 = "  The person is non-diabetic. 😄"
    else:
        res2 = "  The person is diabetic! 😔 "
    
    
    return render(request, 'predict.html' , {'result2':res2})