import pandas as pd

df = pd.read_csv("bitcoin_usd.csv")

df = df.drop(labels=['_id', 'time'], axis=1)
