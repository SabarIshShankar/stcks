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

quandl.ApiConfig.api_key = 'aaaKAWYWyKgsGAjfKzJh'
df = quandl.get("EOD/AAPL")

dataset = df[['Adj_Open','Adj_High','Adj_Low',
'Adj_Close','Adj_Volume']].copy()
dataset = dataset.iloc[pd.np.r_[:,-501,-1]]