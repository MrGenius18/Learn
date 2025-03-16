import pandas as pd
import numpy as np
############################

data = pd.DataFrame()

# Handle missing value ############################
    # fill missing date
complete_idx = pd.date_range(start='2022-01-01', end='2025-12-31', freq='D')
data.reindex(complete_idx)

    # fill missing value
        # using Imputation ==> fillna(mean/mode) / ffill() / bfill()
        # using interpolation ==> liner / spline / polynomial
data.interpolate(method='linear') 
data.interpolate(method='polynomial', order=2) # spline
############################

# Resampling ############################
    # Up Sampling ==> in-creasing freq. -> Daily to Hour
data['close'].resample('H').interpolate(method='linear')
    # Down Sampling ==> de-creasing freq. -> Daily to Month
data['close'].resample('M').mean()
############################

# Making Data Stationary ############################
############################

# Handling Outliers ############################
    # Imputation / interpolation
    # Transformation method (log / power / box-cox)
    # Smoothing method (Moving Average / Exponential)
############################


#### fill missing value using Predictive ML model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()

known_df = data.dropna(subset=['close'])
miss_df = data[data['close'].isna()]

lr_model.fit(known_df[['open']], known_df['close'])
pred_close = lr_model.predict(known_df[['open']])

data.loc[data['close'].isna()] = pred_close