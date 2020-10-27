import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('bitcoin_usd.csv')

print(df)

df = df.drop(labels=['_id', 'time'], axis=1)

plt.plot(df)
plt.show()
