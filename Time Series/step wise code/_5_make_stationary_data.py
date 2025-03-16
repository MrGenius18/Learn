import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
############################

prices = apple_stock_data.Close # data
stock_data = pd.DataFrame() # dummy data

# differencing 1st/2nd order ############################
first_order = prices.diff()
second_order = first_order.diff()
############################

# Transformation (handle outlier) ############################
price_log = np.log(prices) # log
price_sqrt = np.sqrt(prices) # power
price_boxcox, lam = stats.boxcox(prices.values[prices>0]) # only positive value require
############################

# De-tranding ############################
    # using linear trend
trend = np.polyfit(np.arange(len(prices)), prices, 1)
trend_line = np.polyval(trend, np.arange(len(prices)))
price_detrended = prices.values - trend_line

    # using moving average
price_ma = prices.rolling(window=12).mean()
price_detrended = prices - price_ma
price_detrended = price_detrended.dropna()

### plot Visualize
plt.plot(np.arange(len(prices)), trend_line, label='trend_line')
plt.plot(np.arange(len(prices)), price_ma, label='moving_line')
plt.plot(np.arange(len(prices)), prices, label='observed_value')
############################

# Seasonal Adjustment ############################
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(prices, model='additive', period=30)
price_adjusted = prices.values / decomposition.seasonal.values
############################

# Smoothing method (handle outlier) ############################
    # Moving Average Smoothing
window_size = 20

stock_data["SMA"] = stock_data['Apple'].rolling(window=window_size).mean() # Simple MA

weights = np.arange(1, window_size+1)
stock_data["WMA"] = stock_data['Apple'].rolling(window=window_size).apply(lambda price: np.dot(price, weights)/weights.sum(), raw=True) # Weighted MA

stock_data["EMA"] = stock_data['Apple'].ewm(span=window_size).mean() # Exponential MA

    # Exponential Smoothing <- 0 means no noise ==> high smoothing

ses_model = SimpleExpSmoothing(stock_data['Apple']).fit(smoothing_level=0.7) 
stock_data["SES"] = ses_model.fittedvalues # Simple ES

des_model = ExponentialSmoothing(stock_data['Apple'], trend='add').fit(smoothing_level=0.7)
stock_data["DES"] = des_model.fittedvalues # Double EES — Holt's linear trend model

tes_model = ExponentialSmoothing(stock_data['Apple'], trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=0.7)
stock_data["TES"] = tes_model.fittedvalues # Triple ES — Holt's Winters method
############################

stock_data.tail()
adf_test(price_log)