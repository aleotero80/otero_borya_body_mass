# SCRIPT CONTAINING LIBRARIES AND FUNCTION TO CALCULATE BODY MASS IN BIPEDS (RANDOM FOREST)

# LIBRARIES

#!pip install ipywidgets
#!pip install matplotlib
#!pip install seaborn
#!pip install plotly
#!pip install statsmodels
#%pip install pydotplus
#!pip install pandas
#!pip install scikeras
#!pip install shap
#!pip install pillow
#!pip install shap
#!pip install eli5
#!pip install -U kaleido
#!pip install openpyxl
#!pip install imbalanced-learn
#!pip install smogn
#%pip install tensorflow
import tensorflow as tf
import keras
import keras.backend as k
from keras import layers
#!pip install keras-tuner
import keras_tuner as kt
import os
import math
import pandas as pd
#import shap
#import eli5
#import smogn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np
import seaborn as sns
#from plotly.offline import plot
#import plotly.express as px
#import plotly.graph_objects as go
#import plotly.io as pio
#import statsmodels.api as sm
#from sklearn.inspection import permutation_importance
#from eli5.sklearn import PermutationImportance
#from sklearn.model_selection import validation_curve
from keras import regularizers
from sklearn import preprocessing
#from sklearn.preprocessing import normalize
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import RobustScaler
#from sklearn.model_selection import GridSearchCV
#%matplotlib inline
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.svm import SVR
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from sklearn.cluster import KMeans
#import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
#from sklearn import svm
from six import StringIO
from sklearn.tree import export_graphviz
from sklearn.datasets import make_blobs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
#import pydotplus
#from pydotplus import graph_from_dot_data
#from sklearn.datasets import make_blobs
#from imblearn.over_sampling import SMOTE

#----UPLOADING FILES----#

# Upload training datset

df = pd.read_excel('Table_S8_full_train_data.xlsx')

#----BM FUNCTION----#

def bm_random_f():
    # Type available measurements
    FL = float(input('Enter femur length in mm: '))
    FC = float(input('Enter femur circumference in mm: '))
        
    # Convert training data to natural logarithm
    for col in df.select_dtypes(include = np.number).columns:
        df[col] = np.log(df[col])
    
    # Store available measurements in a dictionary linked to training variables
    keys = ['Femur length (mm)', 'Femur circumference (mm)']
    values = [FL, FC]
    measurements = dict(zip(keys, values))
    items_to_remove = list(measurements.keys())
    for key in items_to_remove:
        if measurements[key] == 0:
            del measurements[key]
    columns = [] 
    for key in measurements:
        columns.append(key)
    dictionary = {} # Create empty dictionary
    for col_name, col_series in df.items():
        if col_name in columns:
            dictionary.setdefault(col_name, col_series)
    
    # Store predictors in X and target in y
    X = pd.DataFrame.from_dict(dictionary)
    y = df['Body mass (g)']
    
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
    # Create algorithm instances
    # RFdh
    forest_dh = RandomForestRegressor(random_state = 42)
    # RFoh
    forest_oh = RandomForestRegressor(
        bootstrap = True,
        max_depth = 10,
        max_features = 0.5,
        max_samples = 0.5,
        min_samples_leaf = 1,
        min_samples_split = 4,
        n_estimators = 300,
        random_state = 42,
        oob_score = True,
        n_jobs = -1
    )
        
    # Store models in a list
    models = [forest_dh, forest_oh]
    
    # loop the list and fit each model
    for i in models:
        i.fit(X_train, y_train)
    X_pred = pd.DataFrame([measurements])
    for col in X_pred.select_dtypes(include = np.number).columns:
        X_pred[col] = np.log(X_pred[col])
    
    # Iterate the predictors over algorithms to predict body mass
    for j in models:
        predictions = np.exp(j.predict(X_pred))
        predictions = np.round(predictions, 2)
        upper_bound = np.round(predictions * 1.28, 2) #calculate error upper
        lower_bound = np.round(predictions * 0.72, 2) #calculate error lower
        print('The predictions (in grams) for:\n', j, 'are:\n\n', predictions.tolist())
        print('\n')
        print('The upper and lower error boundaries are:\n', upper_bound.tolist(), 'and', lower_bound.tolist())
        print('\n')
    # End function