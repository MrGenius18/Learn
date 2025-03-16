import matplotlib.pyplot as plt
from sklearn.metrics import *
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
############################

### AR / MA / ARMA / ARIMA / SARIMA ### <== Univariante/single variable
    # AR => Auto-Regressive <-- p
    # I => Integrated <-- d
    # MA => Moving Average <-- q
    # S => Seasonal <-- s

data = first_order.dropna() # data
train_data, test_data = data.iloc[:-30], data.iloc[-30:] # split the data

# AR Model ############################
model = AutoReg(train_data, lags=30).fit() # 30 days prev & future predict
############################

# MA / ARMA / ARIMA Model ############################
model = ARIMA(train_data, order=(7,1,30)).fit() # (p,d,q) set based on the Model
############################

# SARIMA Model ############################
model = SARIMAX(train_data, order=(7, 1, 7), seasonal_order=(1, 1, 1, 45)).fit() # (p,d,q) & (P,D,Q,S)
############################

# Predict, Visualize & Evaluate the Model ############################
predictions = model.predict(start=len(train_data),
                           end=len(train_data)+len(test_data)-1,
                           dynamic=False)

plt.plot(test_data.index, test_data, label="Test data")
plt.plot(test_data.index, predictions, color='red', linestyle='--', label="Predicted data")

print(f"RMSE: {root_mean_squared_error(test_data, predictions):.2f}")
print(model.aic, model.bic)

# predict for next month ############################
model.plot_predict(1, 31) # total_len_record + (1*31) <- for next 1month if 2year(2*365)
model.forecast(steps=31) # array formate not graph
############################


######################
test_start_date = test_data.index[0]
test_end_date = test_data.index[-1]

pred_value = model.predict(start=test_start_date, end=test_end_date)
residuals = test_data['Close']-pred_value
model.resid.plot(kind='kde')

test_data['Pred_value'] = pred_value
######################
