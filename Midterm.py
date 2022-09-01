# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:01:33 2022

@author: brije
"""

import pandas
import numpy

from keras.layers import Dense 
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

#Read the excel file
data = pandas.read_excel('C:/Users/brije/Downloads/MidtermDataset.xlsx')

y = data['Consumption']

x = data[['Income', 'Job type']]

#Train and test split
x_train = x[0:55]
y_train = y[0:55]

x_test = x[55:68]
y_test = y[55:68]

#fit and transform the test and train data
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

#Build the NN model
model = Sequential()

activation1 = 'linear'
activation2 = 'linear'

print('Using activation1: {activation1} and activation2: {activation2}'.format(activation1=activation1, activation2=activation2))
model.add(Dense(100, activation = activation1, input_dim = 2))
model.add(Dense(50, activation = activation2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'] )

# Train the NN model
history = model.fit(x_train_sc, y_train, validation_data=(x_test_sc, y_test), epochs=1000, verbose=0)

y_test_predict = model.predict(x_test_sc, verbose=0)

y_test_predict_reshaped = y_test_predict.reshape(1, 13)
y_test_numpy = y_test.to_numpy()

# Calculate accuracy
accuracy = 1 - numpy.mean(abs(numpy.subtract(y_test_numpy, y_test_predict_reshaped))/y_test_numpy)
print('Accuracy: {accuracy:.2%}'.format(accuracy=accuracy))

# Generate prediction
income = 33000
jobType = 1
print('For income {income}, and ({jobType}):Professional job type consumption prediction is {consumption_prediction:.2f}'
      .format(income=income, jobType=jobType, consumption_prediction=model.predict(sc.transform([[income, jobType]]), verbose=0)[0][0]))
