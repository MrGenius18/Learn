from statsmodels.tsa.seasonal import seasonal_decompose, STL
import matplotlib.pyplot as plt
############################

prices = apple_stock_data.Close # data

# Classical decomposition model using additive/multiplicative ############################
decomposition_additive = seasonal_decompose(prices, model='additive', period=30)

trend_add = decomposition_additive.trend
seasonal_add = decomposition_additive.seasonal
resid_add = decomposition_additive.resid
############################

# STL decomposition model ############################
decomposition_stl = STL(prices, period=30) # model always additive
result = decomposition_stl.fit()
############################

# both visualize ploting ############################
### Classical
plt.subplot(4, 2, 1)
plt.title("Classical Additive Decomposition", fontname='Rockwell')
plt.plot(prices, label='Original')
plt.subplot(4, 2, 3)
plt.plot(trend_add, label='Trend') 
plt.subplot(4, 2, 5)
plt.plot(seasonal_add, label='Seasonal')
plt.subplot(4, 2, 7)
plt.plot(resid_add, label='Residuals')

### STL
plt.subplot(4, 2, 2)
plt.title("STL Decomposition using Loess", fontname='Rockwell') 
plt.plot(result.observed, label='Original', color='orange')
plt.subplot(4, 2, 4)
plt.plot(result.trend, label='Trend', color='orange')
plt.subplot(4, 2, 6)
plt.plot(result.seasonal, label='Seasonal', color='orange')
plt.subplot(4, 2, 8)
plt.plot(result.resid, label='Residuals', color='orange')
############################