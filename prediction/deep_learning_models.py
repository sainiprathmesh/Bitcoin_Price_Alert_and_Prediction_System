import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
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

# Lets Do the prediction and check performance metrics
train_predict = model1.predict(X_train)
test_predict = model1.predict(X_test)
print("Train Predict: ", train_predict)
print("Test Predict: ", test_predict)
print("Execution time: ", eg_time - sg_time)

# Calculate MSE performance metrics
from sklearn.metrics import mean_squared_error

mse0 = mean_squared_error(y_train, train_predict)
print("mse_train_gru: ", mse0)

# Calculate MSE performance metrics
from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(y_test, test_predict)
print("mse_test_gru: ", mse1)

# Calculate r2 performance metrics
from sklearn.metrics import r2_score

r1 = r2_score(y_test, test_predict)
print("r2_gru: ", r1)

# define model2 LSTM
sl_time = time.time()
model2 = Sequential()
model2.add(LSTM(100, activation='linear', return_sequences=True, input_shape=(3, 1)))
model2.add(LSTM(50, activation='linear'))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mse')
# fit model
model2.fit(X_train, y_train, epochs=100, verbose=1)
el_time = time.time()

# Lets Do the prediction and check performance metrics
train_predict = model2.predict(X_train)
test_predict = model2.predict(X_test)
print("Train Predict: ", train_predict)
print("Test Predict: ", test_predict)
print("Execution Time: ", el_time - sl_time)

# Calculate MSE performance metrics
from sklearn.metrics import mean_squared_error

mse_train_lstm = mean_squared_error(y_train, train_predict)
print("mse_train_lstm: ", mse_train_lstm)

# Calculate MSE performance metrics
from sklearn.metrics import mean_squared_error

mse_test_lstm = mean_squared_error(y_test, test_predict)
print("mse_test_lstm: ", mse_test_lstm)

# Calculate r2 performance metrics
from sklearn.metrics import r2_score

r2 = r2_score(y_test, test_predict)
print("r2_lstm: ", r2)

# plot
plt.plot(scaler.inverse_transform(y_test))
plt.plot(scaler.inverse_transform(test_predict))
plt.xlim(200, 300)
plt.show()

df_model = pd.DataFrame({'Model_Applied': ['GRU', 'LSTM'], 'MSE': [mse1, mse_test_lstm], 'R2': [r1, r2],
                         'Execution Time': [eg_time - sg_time, el_time - sl_time]})
print(df_model)
