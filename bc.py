import requests
import pandas as pd
from matplotlib import pyplot as plt

data = "https://api.coindesk.com/v1/bpi/historical/close.json"
r = requests.get(data)

with open("btc.json", "w") as f:
	f.write(r.text)

df = pd.read_json("btc.json").iloc[: , :-2]

df.drop(df.tail(2).index, iplace=True)
df.plot()
plt.tight_layout()
plt.show()

