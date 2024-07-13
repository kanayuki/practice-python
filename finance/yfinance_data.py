import yfinance as yf

msft = yf.Ticker("MSFT")

# Get stock info    
print(msft.info)

# Get historical prices
hist = msft.history(period="1mo")
print(hist)

# Plot historical prices
# hist.plot()

# Get current price 
# print(msft.info["regularMarketPrice"])
