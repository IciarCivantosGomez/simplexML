import random

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
from imblearn.over_sampling import SMOTE
import seaborn as sns
sns.set(color_codes=True)

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error


import rse
verbose = False

print("Estimacion solo con parametros medioambientales")
print("===============================================")

individuals_train = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')

num_rows = len(individuals_train)
num_cols = len(individuals_train.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))

base_list = ['species','individuals','ph','salinity','cl','co3','c','mo','n','cn','p','ca','mg','k','na','precip']
col_list = base_list.copy()
col_list.append('present')
individuals_train = individuals_train[col_list]

individuals_types = individuals_train.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(individuals_train[['species']])
individuals_train[['species']] = le.transform(individuals_train[['species']])

"Transformamos la variable present a numérica"
le = LabelEncoder()
le.fit(individuals_train[['present']])
individuals_train[['present']] = le.transform(individuals_train[['present']])

perc_0s = round(len(np.where(individuals_train[['present']] == 0)[0])/num_rows * 100,2)
perc_1s = round(len(np.where(individuals_train[['present']] == 1)[0])/num_rows * 100,2)

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
    
    sm = SMOTE(random_state=42,sampling_strategy = smote_0s)
    individuals_train, y_res = sm.fit_resample(individuals_train[base_list], individuals_train[['present']])
    individuals_train = individuals_train.join(y_res)
    
else:
    print("===============================================")
    print("No se aplicará SMOTE")
    print("===============================================")

if verbose:
    print(individuals_train.dtypes)


"Feature Importance"

input_features = [column for column in list(individuals_train) if column != 'individuals']
X = individuals_train[input_features]
X['random_noise'] = np.random.normal(size=X.shape[0])
y = individuals_train['individuals']

# RF K-Fold train
classifier = RandomForestRegressor(n_jobs=-1)
cv = cross_validate(estimator=classifier, X=X, y=y, cv=5,return_estimator=True)

feature_importance = {}
for k in range(0, len(cv['estimator'])):
    feature_importance['k_{}'.format(k + 1)] = cv['estimator'][k].feature_importances_
feature_importance = pd.DataFrame(feature_importance, index=X.columns)

feature_importance = feature_importance.mean(axis=1).to_frame('importance') \
        .sort_values('importance', ascending=False)
feature_selection = feature_importance.to_dict()

# Get importance concentration score
importance_concentration = (feature_importance.iloc[1] / feature_importance.iloc[0]).values[0] < 0.10
feature_selection['importance_concentration'] = {'assessment': importance_concentration,
                                                 'score': 1 if importance_concentration else 3}

# Get selected features score
selected_features = feature_importance['importance'] > feature_importance.loc['random_noise'].values[0]
selected_features = feature_importance.index[selected_features].tolist()

feature_importance.reset_index(inplace = True)



"Estandarizacion de los datos"

variables_to_ignore = ['individuals']
selected_features = [element for element in list(individuals_train) if element not in variables_to_ignore]

individuals_model_train = individuals_train[selected_features]

std_scaler = StandardScaler()

std_scaler_model = std_scaler.fit(individuals_model_train)
individuals_model_train = std_scaler_model.transform(individuals_model_train)

"Division Train Test"

X = pd.DataFrame(data = individuals_model_train, columns = selected_features)
y = individuals_train.individuals


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8)
print(X_train.columns)

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