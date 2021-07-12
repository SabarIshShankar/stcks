import dash
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output, Event
import plotly
import plotly.plotly as py 
import plotly.graph_objs as go 
from plotly import tools
import numpy as np 
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import datetime
import csv
import quandl, math 

from sklearn import preprocessing, cross_validation, svm
from sklearn.svm import SVR
from sklearn.linear_model import LinearReagression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ARDRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from matplotlib import style
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


quandl.ApiConfig.api_key = 'aaaKAWYWyKgsGAjfKzJh'
df = quandl.get("EOD/AAPL")

dataset = df[['Adj_Open','Adj_High','Adj_Low',
'Adj_Close','Adj_Volume']].copy()
dataset = dataset.iloc[pd.np.r_[:,-501:-1]]

dataset['HL_PCT'] = (dataset['Adj_High'] - dataset['Adj_Low'])/dataset['Adj-Low'] * 100.0
dataset['PCT_change'] = (dataset['Adj_Close'] - dataset['Adj_Open'])/dataset['Adj_Open'] * 100.0

dataset = dataset[['Adj_CLose','HL_PCT','PCT_change', 'Adj_Volume']]

pred_feature = 'Adj_CLose'
dataset.fillna(value = 99999, inplace=True)
no_of_var = int(math.ceil(0.1 * len(dataset)))

dataset['label'] = dataset[pred_feature].shift(-no_of_var)

x=np.arry(dataset.drop(['label',1]))
x=preprocessing.scale(x)
x_small = x[-no_of_var:]
x_small = x[-no_of_var:]
x = x[:-no_of_var]

dataset.dropna(inplace = True)
y = np.array(dataset['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)
dataset_2 = df[['Adj_Open','Adj_High',  'Adj_Low',  'Adj_Close', 'Adj_Volume']].copy()
dataset_2 = dataset_2.iloc[pd.np.r_[:,-51:-1]]
df = df.iloc[pd.np.r_[:,-801:-1]]

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
net_unix = last_unix + one_day


model1 = svm.LinearSVR()
model1.fit(x_train, y_train)
confidence = model1.score(x_test, y_test)
predict_1 = model1.predict(x_small)
dataset['Predict_Linear'] = np.nan
print('Score for Linear Reggression', confidence1)

for i in predict_1:
	next_data = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	dataset.loc[next_date] = [np.nan for _ in range(len(dataset.cloumns)-1)]+[i]