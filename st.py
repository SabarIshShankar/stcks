import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plot
start = datetime.datetime(2021,6,1)
end = datetime.datetime(2021,6,9)
n = web.DataReader("NFLX",'yahoo',start,end)
n['Open'].plot()
n['Close'].plot()
plot.show()
