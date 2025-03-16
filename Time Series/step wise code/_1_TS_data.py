# download any stock related data
import yfinance as yf
import matplotlib.pyplot as plt
############################

# download apple & tesla stock price data ############################
apple_stock_data = yf.download('AAPL', period='1y')
tesla_stock_data = yf.download('TSLA', period='1y')
apple_stock_data.head()

apple_prices = apple_stock_data.Close
tesla_prices = tesla_stock_data.Close
apple_prices.head(3)
############################

# data visualize ############################
plt.figure(figsize=(12, 5))

plt.plot(apple_prices.index, apple_prices, label='Apple Closing Price')

plt.title("AAPL Stock Price Over Time", fontname='Rockwell', fontsize=20)
plt.xlabel("Date")
plt.ylabel("Price (USD)")

plt.legend()
plt.grid(True)
plt.show()
############################