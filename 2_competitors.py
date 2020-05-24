# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:17:02 2020

@author: iciar
"""

import pandas as pd
import random
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from imblearn.over_sampling import SMOTE, RandomOverSampler

import rse
verbose = False

print("Predictor en dos pasos")
print("======================")

environment_train = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
competitors_train = pd.read_csv('datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')

conditions = environment_train.merge(competitors_train)
 

num_rows = len(conditions)
num_cols = len(conditions.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))


col_list = ['species', 'individuals',
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'present',
       'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

train_list = ['species', 'individuals', 
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'present']

conditions = conditions[col_list]

conditions_types = conditions.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(conditions[['species']])
conditions[['species']] = le.transform(conditions[['species']])


"Transformamos la variable present a numérica"
le = LabelEncoder()
le.fit(conditions[['present']])
conditions[['present']] = le.transform(conditions[['present']])

sm = SMOTE(random_state=42)
conditions, y_res = sm.fit_resample(conditions[['species', 'individuals',
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip', 'present',
       'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']], conditions[['present']])


"Estandarizacion de los datos"

conditions_model_train = conditions[train_list]

std_scaler = StandardScaler()
std_scaler_model = std_scaler.fit(conditions_model_train)
conditions_model_train = std_scaler_model.transform(conditions_model_train)

y_pred = {}
rmse_rf = {}
rse_rf = {}
features_to_pred = ['BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']

X = pd.DataFrame(data = conditions_model_train, columns = train_list)
y = conditions[features_to_pred]

X_train_species, X_test_species, y_train_species, y_test_species = train_test_split(X, y, train_size= 0.8)


for i in range(0, len(features_to_pred)):

    variables_to_ignore = features_to_pred[i]
    print("--------------TARGET "+str(variables_to_ignore))
    
    "Division Train Test"
    
    
    X_train = X_train_species
    y_train = y_train_species[variables_to_ignore]
    
    X_test = X_test_species
    y_test = y_test_species[variables_to_ignore]
    
        
    "Algoritmos y Evaluación"
    
    "Random Forest"


    rf = RandomForestRegressor(n_jobs = -1)
    
    rf.fit(X_train,y_train)
    predictions_rf = rf.predict(X_test)
    
    y_pred[variables_to_ignore] = predictions_rf
    rmse_rf[variables_to_ignore] = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))
    mse_rf = mean_squared_error(y_test,predictions_rf)
    rse_rf = rse.calc_rse(predictions_rf,mse_rf)
    #print("RMSE: "+str(rmse_rf[variables_to_ignore]))
    print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf[variables_to_ignore],rse_rf))




"Utilizamos los resultados para predecir individuals"

features_to_pred = ['individuals']
selected_features = [element for element in col_list if element not in features_to_pred]

new_X = X_test_species.reset_index().drop(['index'], axis = 1)
y_predictions = pd.DataFrame.from_dict(y_pred)
X_individuals = new_X.join(y_predictions)[selected_features]

y_individuals = conditions[features_to_pred].iloc[y_test_species.index].reset_index().drop(['index'], axis = 1)

data = X_individuals.join(y_individuals)

sm = SMOTE(random_state=42,sampling_strategy = {-1: 5710, 1: 28550})
data, y_res = sm.fit_resample(data[['species', 'individuals',
       'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
       'k', 'na', 'precip',
       'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
       'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
       'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']], data[['present']])
data = data.join(y_res)

X_ind = data[selected_features]
y_ind = data['individuals']

X_train_individuals, X_test_individuals, y_train_individuals, y_test_individuals = train_test_split(X_ind, y_ind, train_size= 0.8)

"Random Forest"
print("Random Forest")
seed_value = 4
random.seed(seed_value)

rf = RandomForestRegressor(random_state= seed_value, n_jobs = -1, n_estimators = 150)
rf.fit(X_train_individuals,y_train_individuals)
predictions_rf = rf.predict(X_test_individuals)


rmse_rf_final = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_rf))
mse_rf_final = mean_squared_error(y_test_individuals,predictions_rf)
rse_rf_final = rse.calc_rse(predictions_rf,mse_rf_final)
print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf_final,rmse_rf_final,rse_rf_final))




    
