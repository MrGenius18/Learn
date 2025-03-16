import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR, VARMAX
############################

### VAR / VMA / VARMA / VARIMA ### <== Multivariante/multiple variable
    # AR => Auto-Regressive <-- p
    # I => Integrated <-- d
    # MA => Moving Average <-- q
    # V => Vector <-- other factor corr.

stock_data = pd.DataFrame()

stock_data['Tesla'] = tesla_prices
stock_data['Apple'] = apple_prices.shift() # yesterday data based predict today tesla data
stock_data.dropna(inplace=True)

# Granger Causality Test ############################
grangercausalitytests(stock_data, maxlag=[14])
print() # P-value < 0.05 ==> Tesla stock closing price affected by Apple closing price
############################

data = stock_data.diff().dropna() # data
train_data, test_data = data[:-14], data[-14:] # split the data

# VAR Model ############################
model = VAR(train_data).fit(maxlags=7)

predictions = model.forecast(train_data.values[-model.k_ar:], steps=len(test_data))
predictions = pd.DataFrame(predictions, index=test_data.index, columns=test_data.columns)
############################

# VMA / VARMA ############################
model = VARMAX(train_data, order=(0, 14)).fit() # (p,q) set based on the Model

predictions = model.predict(start=len(train_data),
                           end=len(train_data)+len(test_data)-1,
                           dynamic=False)
############################

# Visualize & Evaluate the Model ############################
plt.plot(test_data.index, test_data['Tesla'], label="Actual price", color='blue')
plt.plot(test_data.index, predictions['Tesla'], label="Predicted price", color='orange')

print("RMSE:", root_mean_squared_error(test_data['Tesla'], predictions['Tesla']))
print(model.aic, model.bic)
############################

# predict for next month ############################
model.plot_predict(1, 31) # total_len_record + (1*31) <- for next 1month if 2year(2*365)
model.forecast(steps=31) # array formate not graph
############################