from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
############################

prices = apple_stock_data.Close # data

# week stationary data test (ADF & KPSS) ############################
def adf_test(series):
    result = adfuller(series)

    print("Stationay Rules: P_value < 0.05 || Statistics_value > Critical_value\n")
    print("ADF Test Results: Statistics:", round(result[0], 3), end="  ")
    print("P-value:", result[1])
    print("Critical Values: ")
    for key, val in result[4].items():
        print(f"    {key} : {val:.3f}")

def kpss_test(series):
    result = kpss(series, regression='ct') # data is stationary around constant/trend

    print("Stationay Rules: P_value > 0.05 || Statistics_value < Critical_value\n")
    print("KPSS Test Results: Statistics:", round(result[0], 3), end="  ")
    print("P-value:", result[1])
    print("Critical Values: ")
    for key, val in result[3].items():
        print(f"    {key} : {val}")

adf_test(prices)
kpss_test(prices)
############################

# streek stationary data test (KS-test) ############################
def ksTestStationarity(series):
    split_len = len(series) // 2
    first_half_series = series[:split_len]
    second_half_series = series[split_len:]

    stat, p_val = ks_2samp(first_half_series, second_half_series)

    print("Streek Stationay Rules: P_value > 0.05")

    ### visualize plot
    plt.plot(series, color='blue')
    plt.title(f"Strict Stationarity Check: P-value:{p_val:.5f}")
    plt.show()

ksTestStationarity(prices)
############################