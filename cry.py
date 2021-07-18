import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt


currency = "USD"
metric="Close"
start = dt.datetime(2015,1,1)
end=dt.datetime.now()
crypto =  ['BTC','ETH','LTC','DOGE','DINK']
names = []

first = True
for ticker in crypto:
    data=web.DataReader(f'{ticker}-{currency}', "yahoo", start, end)
    if  first:
        combined = data[[metric]].copy()
        names.append(ticker)
        combined.columns = names
        first = False
    else:
        combined = combined.join(data[metric])
        names.append(ticker)
        combined.columns = names

plt.yscale('log')
for ticker in crypto: 
	plt.plot(combined[ticker], label=ticker)

plt.legend(loc = "upper right")
plt.show()