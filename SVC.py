# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 23:02:21 2022

@author: brijeshkumar amin - Student ID - 2109693
"""

import pandas as pd
import sklearn.svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Read data from the excel file
df = pd.read_csv ('C:/Users/brije/OneDrive/Sem 3/BUSI/collisions_cleaned1.csv' )


df.info()

df = df.sample(n=20000)
df=df.dropna(subset=['COLLISIONTYPE'])

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
df.drop('SDOT_COLDESC', axis=1, inplace=True)
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
df.drop('MONTH', axis=1, inplace=True)
df.drop('YEAR', axis=1, inplace=True)
df.drop('ST_COLDESC', axis=1, inplace=True)
df.drop('SDOT_COLCODE', axis=1, inplace=True)

df['ADDRTYPE']=df['ADDRTYPE'].fillna(df['ADDRTYPE'].mode()[0])
df['JUNCTIONTYPE']=df['JUNCTIONTYPE'].fillna(df['JUNCTIONTYPE'].mode()[0])
df['UNDERINFL']=df['UNDERINFL'].fillna(df['UNDERINFL'].mode()[0])
df['WEATHER']=df['WEATHER'].fillna(df['WEATHER'].mode()[0])
df['ROADCOND']=df['ROADCOND'].fillna(df['ROADCOND'].mode()[0])
df['LIGHTCOND']=df['LIGHTCOND'].fillna(df['LIGHTCOND'].mode()[0])


df['INATTENTIONIND']=df['INATTENTIONIND'].fillna('N')

labelencoder = LabelEncoder()

df['COLLISIONTYPE_CODE'] = labelencoder.fit_transform(df['COLLISIONTYPE'])

#df.info()

df.to_csv('collisions_cleaned_python.csv', index=False)

# Extract x and y columns 
x = df[['ADDRTYPE','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT',
        'JUNCTIONTYPE','UNDERINFL','WEATHER','ROADCOND',
        'LIGHTCOND','INATTENTIONIND']]
y = df['COLLISIONTYPE_CODE']


# Create a transformer to transform the vlaues of Business Type column using OneHotEncoder
transformer = ColumnTransformer(transformers=[('tnf1', OneHotEncoder(sparse=False, drop='first'), 
                                                ['ADDRTYPE','JUNCTIONTYPE','WEATHER','ROADCOND',
                                                 'LIGHTCOND','UNDERINFL','INATTENTIONIND'])], 
                                remainder='passthrough')

# Transform the training and test datasets
x_transformed = transformer.fit_transform(x)

# Produce training and testing datasets
x_train,x_test,y_train,y_test = train_test_split(x_transformed, y, test_size=0.2, random_state=1)

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear']}

grid = GridSearchCV(sklearn.svm.SVC(),param_grid,refit=True,verbose=2, cv=2)

grid.fit(x_train, y_train)

print(grid.best_estimator_)

predicted_y_values = grid.predict(x_test)

print(confusion_matrix(y_test,predicted_y_values))
print(classification_report(y_test,predicted_y_values))
