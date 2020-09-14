# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:17:02 2020

@author: iciar
"""

import pandas as pd
import random
pd.set_option('display.max_colwidth', -1)
import numpy as np
import math
import seaborn as sns
sns.set(color_codes=True)

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE, SVMSMOTE

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



perc_0s = round(len(np.where(conditions[['present']] == 0)[0])/num_rows * 100,2)
perc_1s = round(len(np.where(conditions[['present']] == 1)[0])/num_rows * 100,2)

print("===============================================")
print("Proporción inicial especies presentes: "+str(perc_0s)+"% de 0s")
print(" y "+str(perc_1s)+"% de 1s")
print("===============================================")

print("===============================================")
smote_yn = str(input("¿Desea incrementar el %?(y/n): "))

if smote_yn == 'y':
    print("Inserte nuevo porcentaje")
    perc_0s = float(input("Introduzca porcentaje de 1s: "))
    smote_0s = round(perc_0s/100,2)
    
    # sm = SMOTE(random_state=42,sampling_strategy = smote_0s)
    sm = SVMSMOTE(random_state=42,sampling_strategy = smote_0s)
    conditions, y_res = sm.fit_resample(conditions[['species', 'individuals',
        'ph', 'salinity', 'cl', 'co3', 'c', 'mo', 'n', 'cn', 'p', 'ca', 'mg',
        'k', 'na', 'precip',
        'BEMA', 'CETE', 'CHFU', 'CHMI', 'COSQ', 'FRPU', 'HOMA', 'LEMA', 'LYTR',
        'MEEL', 'MEPO', 'MESU', 'PAIN', 'PLCO', 'POMA', 'POMO', 'PUPA', 'RAPE',
        'SASO', 'SCLA', 'SOAS', 'SPRU', 'SUSP']], conditions[['present']])
    conditions = conditions.join(y_res)
    
else:
    print("===============================================")
    print("No se aplicará SMOTE")
    print("===============================================")



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
X[['present']] = conditions[['present']]
y = conditions[features_to_pred]

X_train_species, X_test_species, y_train_species, y_test_species = train_test_split(X, y, train_size= 0.8, random_state = 0)


"Parámetros Random Forest"

# Number of trees in random forest
n_estimators = [100,150]
# Number of features to consider at every split
max_features = ['auto']
#Grid Search
random_grid = {'n_estimators': n_estimators,
           'max_features': max_features}



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
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)
    
    rf_random.fit(X_train,y_train)
    print(rf_random.best_params_)
    predictions_rf = rf_random.best_estimator_.predict(X_test)
    
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
y_predictions = y_predictions.applymap(lambda x: math.floor(x))
X_individuals = new_X.join(y_predictions)[selected_features]

y_individuals = conditions[features_to_pred].iloc[y_test_species.index].reset_index().drop(['index'], axis = 1)

data = X_individuals.join(y_individuals)
    
sm = SMOTE(random_state=42)
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

rf = RandomForestRegressor(n_jobs = -1)
# rf = RandomForestRegressor(random_state= seed_value, n_jobs = -1, n_estimators = 150)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 7, verbose=2, n_jobs = -1)

rf_random.fit(X_train,y_train)
print(rf_random.best_params_)
predictions_rf = rf_random.best_estimator_.predict(X_test)

rmse_rf_final = np.sqrt(metrics.mean_squared_error(y_test_individuals, predictions_rf))
mse_rf_final = mean_squared_error(y_test_individuals,predictions_rf)
rse_rf_final = rse.calc_rse(predictions_rf,mse_rf_final)
print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf_final,rmse_rf_final,rse_rf_final))




    
