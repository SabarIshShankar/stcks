import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plot
import sub
start = datetime.datetime(2021,4,6)
end = datetime.datetime(2021,5,5)
g = web.DataReader("GOOGL",'yahoo',start,end)
g['Close'].plot()
plot.show()
sub.data