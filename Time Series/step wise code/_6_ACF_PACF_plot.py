from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
############################

prices = apple_stock_data.Close # data

# ACF plot
plt.subplot(1, 2, 1)
plot_acf(prices.diff().diff().dropna(), ax=plt.gca(), lags=50)
plt.title("ACF of APPL stock")

# PACF plot
plt.subplot(1, 2, 2)
plot_pacf(prices.diff().diff().dropna(), ax=plt.gca(), lags=50, method='ywm')
plt.title("PACF of APPL stock")

plt.tight_layout()
plt.show()

### ACF ==> cut-off ==> MA(q) !== decay exponentially/sinusoidly
### PACF ==> cut-off ==> AR(p) !== decay exponentially/sinusoidly

### All under Confidence Interval ==> White Noice data === Not Predictable