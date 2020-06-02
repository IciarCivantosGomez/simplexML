
import random

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
import seaborn as sns
sns.set(color_codes=True)

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

import rse
verbose = False

print("Estimacion solo con parametros medioambientales")
print("===============================================")

individuals = pd.read_csv('datasets/abund_merged_dataset_onlyenvironment.csv', sep=',')
#competition_test = pd.read_csv('datasets/competition_merged_dataset_2019.csv', sep=',')

num_rows = len(individuals)
num_cols = len(individuals.columns)
print("This dataset has {0} rows and {1} columns".format(num_rows, num_cols))

base_list = ['year','species','individuals','ph','salinity','cl','co3','c','mo','n','cn','p','ca','mg','k','na','precip']
col_list = base_list.copy()
col_list.append('present')
individuals = individuals[col_list]
#competition_test = competition_test[col_list]

individuals_types = individuals.dtypes

"Data Wrangling"

"Transformamos la variable species a numérica"
le = LabelEncoder()
le.fit(individuals[['species']])
individuals[['species']] = le.transform(individuals[['species']])

"Transformamos la variable present a numérica"
le = LabelEncoder()
le.fit(individuals[['present']])
individuals[['present']] = le.transform(individuals[['present']])

if verbose:
    print(individuals.dtypes)

"Feature Importance"

input_features = [column for column in list(individuals) if column != 'individuals']
X = individuals[input_features]
X['random_noise'] = np.random.normal(size=X.shape[0])
y = individuals['individuals']

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