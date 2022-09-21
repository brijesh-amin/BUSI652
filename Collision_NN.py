# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 23:02:21 2022

@author: brijeshkumar amin - Student ID - 2109693
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from keras.layers import Dense 
from keras.models import Sequential
from matplotlib import pyplot

# Read data from the excel file
df = pd.read_csv ('C:/Users/brije/Downloads/Collisions.csv (1)/Collisions.csv' )

df.info()

df.drop('X', axis=1, inplace=True)
df.drop('Y', axis=1, inplace=True)
df.drop('OBJECTID', axis=1, inplace=True)
df.drop('REPORTNO', axis=1, inplace=True)
df.drop('STATUS', axis=1, inplace=True)
df.drop('INTKEY', axis=1, inplace=True)
df.drop('LOCATION', axis=1, inplace=True)
df.drop('EXCEPTRSNCODE', axis=1, inplace=True)
df.drop('EXCEPTRSNDESC', axis=1, inplace=True)
df.drop('SEVERITYCODE', axis=1, inplace=True)
df.drop('SEVERITYDESC', axis=1, inplace=True)
df.drop('INJURIES', axis=1, inplace=True)
df.drop('SERIOUSINJURIES', axis=1, inplace=True)
df.drop('FATALITIES', axis=1, inplace=True)
df.drop('INCDATE', axis=1, inplace=True)
df.drop('INCDTTM', axis=1, inplace=True)
df.drop('PEDROWNOTGRNT', axis=1, inplace=True)
df.drop('SDOTCOLNUM', axis=1, inplace=True)
df.drop('SPEEDING', axis=1, inplace=True)
df.drop('ST_COLCODE', axis=1, inplace=True)
df.drop('SEGLANEKEY', axis=1, inplace=True)
df.drop('CROSSWALKKEY', axis=1, inplace=True)
df.drop('HITPARKEDCAR', axis=1, inplace=True)

df['ADDRTYPE']=df['ADDRTYPE'].fillna(df['ADDRTYPE'].mode()[0])
df['JUNCTIONTYPE']=df['JUNCTIONTYPE'].fillna(df['JUNCTIONTYPE'].mode()[0])
df['UNDERINFL']=df['UNDERINFL'].fillna(df['UNDERINFL'].mode()[0])
df['WEATHER']=df['WEATHER'].fillna(df['WEATHER'].mode()[0])
df['ROADCOND']=df['ROADCOND'].fillna(df['ROADCOND'].mode()[0])
df['LIGHTCOND']=df['LIGHTCOND'].fillna(df['LIGHTCOND'].mode()[0])

df['INATTENTIONIND']=df['INATTENTIONIND'].fillna('N')

df.info()

#df = df.sample(n=5000)

labelencoder = LabelEncoder()

df['COLLISIONTYPE_CODE'] = labelencoder.fit_transform(df['COLLISIONTYPE'])

print(df.nunique())

# Extract x and y columns 
x = df[['ADDRTYPE','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT',
        'JUNCTIONTYPE','UNDERINFL','WEATHER','ROADCOND',
        'LIGHTCOND','INATTENTIONIND']]
y = df[['COLLISIONTYPE_CODE']]

transformer2 = ColumnTransformer(transformers=[('tnf2', OneHotEncoder(sparse=False), 
                                                ['COLLISIONTYPE_CODE'])], 
                                remainder='passthrough')
# Transform y
y = transformer2.fit_transform(y)

print(transformer2.get_feature_names())

# Create a transformer to transform the vlaues of Business Type column using OneHotEncoder
transformer = ColumnTransformer(transformers=[('tnf1', OneHotEncoder(sparse=False), 
                                                ['ADDRTYPE','JUNCTIONTYPE','WEATHER','ROADCOND',
                                                 'LIGHTCOND','UNDERINFL','INATTENTIONIND'])], 
                                remainder='passthrough')
# Transform x
x = transformer.fit_transform(x)

print(transformer.get_feature_names())

# Produce training and testing datasets
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#Build the NN model
model = Sequential()

activation1 = 'relu'
activation2 = 'relu'

print('Using activation1: {activation1} and activation2: {activation2}'.format(activation1=activation1, activation2=activation2))
model.add(Dense(500, activation = activation1, input_dim = 50))
model.add(Dense(100, activation = activation2))
model.add(Dense(11, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

# Train the NN model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=0)

y_test_predict = model.predict(x_test, verbose=0)

score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
