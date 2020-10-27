import pandas as pd
import seaborn as sns

df = pd.read_csv("bitcoin_usd.csv")

df = df.drop(labels=['_id', 'time'], axis=1)

sns.pairplot(df)
