# SCRIPT CONTAINING LIBRARIES AND FUNCTION TO CALCULATE BODY MASS IN BIPEDS (NEURAL NETWORKS)

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

def bm_neural_n():
    # Type available measurements
    FL = float(input('Enter femur length in mm: '))
    FC = float(input('Enter femur circumference in mm: '))
       
    # Set random seed for reproductivility
    keras.utils.set_random_seed(42)
    
    # Store available measurements in a dictionary linked to training variables
    keys = ['Femur length (mm)', 'Femur circumference (mm)']
    values = [FL, FC]
    input_shape = 2
    measurements = dict(zip(keys, values))
    items_to_remove = list(measurements.keys())
    for key in items_to_remove:
        if measurements[key] == 0:
            del measurements[key]
            input_shape = 1
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
    
    # Convert all numerical variables to natural logarithm
    # log-transform predictors
    X_train_log = np.log(X_train)
    X_test_log  = np.log(X_test)
    
    # log-transform target
    y_train_log = np.log(y_train)
    y_test_log  = np.log(y_test)
    
    # Normalization via StandardScaler
    X_scaler = StandardScaler()
    X_train_s = X_scaler.fit_transform(X_train_log)
    X_test_s  = X_scaler.transform(X_test_log)
    
    # Convert target to dataframe and normalized
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train_log.values.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test_log.values.reshape(-1, 1)).ravel()
    
    # Create algorithm instances
    # neural networks
    # Define the Sequential model
    # Experiment with deeper and wider networks
    # Clear the session to enable re-run the model several times
    k.clear_session()
    model = keras.Sequential()
    
    # Input shape equals the variable dimensions of the dataset
    model.add(Input(shape = (input_shape,))) # input = 1 for one predictor; input = 2 for two predictors
      
    # Internal layers with L2 regularization
    model.add(layers.Dense(20, kernel_regularizer = regularizers.l2(0.001), activation = 'relu'))
    model.add(layers.Dense(20, kernel_regularizer = regularizers.l2(0.001), activation = 'relu'))
    model.add(layers.Dense(1))
    
    # Model Compile
    optimizer = keras.optimizers.Adam(learning_rate = 0.001)
    
    model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['mae', 'mse', 'r2_score'])
    
    # Model training and fitting
    # It is important to store the fitting into a variable in order to visualize the history after.
    history = model.fit(X_train_s, y_train_s, validation_data = (X_test_s, y_test_s), validation_split = 0.25, epochs = 100)
    
    # Prediction on test data
    y_pred_test = model.predict(X_test_s)
    
    # Retrotransformation
    # back to log(BM)
    y_pred_log = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    
    # back to BM
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test_log) # the same as y_test
    
    
    # Convert all numerical variables to natural logarithm
    X_val = pd.DataFrame([measurements])
    X_val_log = np.log(X_val)
    
    # Normalization (DO NOT REFIT)
    X_val_s = X_scaler.transform(X_val_log)
    
    # Prediction new value
    df_val_pred = model.predict(X_val_s)
    
    # Undo standarization - back to Log
    y_new_log = y_scaler.inverse_transform(df_val_pred.reshape(-1, 1)).ravel()
    
    # Retrotransform - back to BM
    y_val_pred = np.exp(y_new_log) # predicted values (in grams)
    #y_val_true = np.exp(y_val_log) # real values (in grams) ---> the original y_val
      
    # Presentation of the results
    predictions = float(y_val_pred)
    predictions = round(predictions, 2)
    upper_bound = np.round(predictions * 1.27, 2)
    lower_bound = np.round(predictions * 0.73, 2)
    print('The predictions (in grams) for Neural networks are: \n\n' , predictions)
    print('\n')
    print('The upper and lower error boundaries are:\n', upper_bound.tolist(), 'and', lower_bound.tolist())
    print('\n')
    
    # End function