import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
api_key='Y6DGMXFDIW6CX72M'
ts= TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_intraday(symbol='TSLA', interval='5min', outputsize='full')
print(data)
close_data = data['close']
change=close_data.pct_change()
print(change)

