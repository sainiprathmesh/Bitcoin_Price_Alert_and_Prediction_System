import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
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

# Multivariate Linear Regression

sm_time = time.time()
model1 = LinearRegression()
model1.fit(X_train, y_train)
em_time = time.time()
print("Train Accuracy: ", model1.score(X_train, y_train))
print("Test Accuracy: ", model1.score(X_test, y_test))
print("Execution Time: ", em_time - sm_time)
