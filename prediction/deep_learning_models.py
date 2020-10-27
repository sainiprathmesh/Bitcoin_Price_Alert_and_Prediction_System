import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential

df = pd.read_csv('bitcoin_usd.csv')

y = df['high']
y = np.array(y)
print(y)

df = df.drop(labels=['_id', 'time'], axis=1)

plt.plot(df)
plt.show()

sns.pairplot(df)

df.corr()

sns.heatmap(df.corr())

df = df.drop(labels=['high'], axis=1)

X = np.array(df)

scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X.reshape(-1, 3))

print(X1.shape)

n_features = 1
X1 = X1.reshape((X1.shape[0], X1.shape[1], n_features))

print(X1.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
y1 = scaler.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2)

print(X_train.shape), print(y_train.shape)

print(X_test.shape), print(y_test.shape)

# define model1 GRU
sg_time = time.time()
model1 = Sequential()
model1.add(GRU(100, activation='linear', return_sequences=True, input_shape=(3, 1)))
model1.add(GRU(50, activation='linear'))
model1.add(Dense(1))
model1.compile(optimizer='adam', loss='mse')

# fit model
model1.fit(X_train, y_train, epochs=100, verbose=1)
eg_time = time.time()

### Lets Do the prediction and check performance metrics
train_predict = model1.predict(X_train)
test_predict = model1.predict(X_test)
print("Train Predict: ", train_predict)
print("Test Predict: ", test_predict)
print("Execution time: ", eg_time - sg_time)

# Calculate MSE performance metrics
from sklearn.metrics import mean_squared_error

mse0 = mean_squared_error(y_train, train_predict)
print("mse0: ", mse0)

# Calculate MSE performance metrics
from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(y_test, test_predict)
print("mse1: ", mse1)

# Calculate r2 performance metrics
from sklearn.metrics import r2_score

r1 = r2_score(y_test, test_predict)
print("r1: ", r1)
