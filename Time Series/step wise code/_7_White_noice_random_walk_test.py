from statsmodels.stats.diagnostic import acorr_ljungbox

prices = apple_stock_data.Close # data

# using L-Jungbox test ############################
lb_test = acorr_ljungbox(prices, lags=[10], return_df=True)
print("P-value < 0.05 ==> Random Walk data")
print("P-value > 0.05 ==> White Noise data\n")
print(lb_test)
############################

# also use ACF-PACF test