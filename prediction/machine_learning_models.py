import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("bitcoin_usd.csv")

df = df.drop(labels=['_id', 'time'], axis=1)

print(sns.pairplot(df))

print(df.corr())
print(sns.heatmap(df.corr()))

print(plt.plot(df))

X = df.loc[:, ['close', 'low', 'open']]
Y = df.loc[:, 'high']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
