import random

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set(color_codes=True)

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost
import rse

import warnings
warnings.filterwarnings("ignore")

verbose = False

print("Predictor solo con especies competidoras")
print("========================================")
competitors_train = pd.read_csv('datasets/abund_merged_dataset_onlycompetitors.csv', sep=',')
environment_train = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
individuals = environment_train.merge(competitors_train)

num_rows = len(individuals)
num_cols = len(individuals.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))


"Rename co3.x column to co3"
#individuals_train.rename(columns={"co3.x": "co3"}, inplace = True)

col_list = ['year','species','present','individuals','BEMA','CETE','CHFU','CHMI','COSQ','FRPU','HOMA','LEMA','LYTR','MEEL','MEPO','MESU','PAIN','PLCO','POMA','POMO','PUPA','RAPE','SASO','SCLA','SOAS','SPRU','SUSP']


individuals_types = individuals.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(individuals[['species']])
individuals[['species']] = le.transform(individuals[['species']])

le = LabelEncoder()
le.fit(individuals[['present']])
individuals[['present']] = le.transform(individuals[['present']])

sm = SMOTE(random_state=42,sampling_strategy = {0: 28188, 1: 140940})
individuals, y_res = sm.fit_resample(individuals[['year','species','individuals','BEMA','CETE','CHFU','CHMI','COSQ','FRPU','HOMA','LEMA','LYTR','MEEL','MEPO','MESU','PAIN','PLCO','POMA','POMO','PUPA','RAPE','SASO','SCLA','SOAS','SPRU','SUSP']], individuals[['present']])
individuals = individuals.join(y_res)

if verbose:
    print(individuals.dtypes)

"Ver si hay registros duplicados"

if verbose:
    print("Ver si hay registros duplicados")
num_rows = len(individuals)
num_cols = len(individuals.columns)

individuals = individuals.drop_duplicates()
num_rows_clean = len(individuals)
if verbose:
    print("In this dataset there were {} repeated records".format(num_rows - num_rows_clean))



"Estandarizacion de los datos"

variables_to_ignore = ['individuals','year']
selected_features = [element for element in list(individuals) if element not in variables_to_ignore]


individuals_model_train = individuals.loc[individuals['year'].isin([2015,2016,2017,2018])][selected_features]
individuals_model_test = individuals.loc[individuals['year'] == 2019][selected_features]

std_scaler = StandardScaler()

std_scaler_model = std_scaler.fit(individuals_model_train)
individuals_model_train = std_scaler_model.transform(individuals_model_train)

std_scaler_model = std_scaler.fit(individuals_model_test)
individuals_model_test = std_scaler_model.transform(individuals_model_test)


"Division Train Test"

X_train = pd.DataFrame(data = individuals_model_train, columns = selected_features)
y_train = individuals.loc[individuals['year'].isin([2015,2016,2017,2018])]["individuals"]

X_test = pd.DataFrame(data = individuals_model_test, columns = selected_features)
y_test = individuals.loc[individuals['year'] == 2019]["individuals"]


"Algoritmos y Evaluación"



"Random Forest"

print("Random Forest")
seed_value = 4
random.seed(seed_value)

regr = RandomForestRegressor(random_state= seed_value, n_jobs = -1, n_estimators = 150)
regr.fit(X_train, y_train)
predictions_rf = regr.predict(X_test)

rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, predictions_rf))

mse_rf = mean_squared_error(y_test,predictions_rf)
rse_rf = rse.calc_rse(y_test,mse_rf)

print("mse {:.4f} rmse {:.4f} rse {:.4f}".format(mse_rf,rmse_rf,rse_rf))

