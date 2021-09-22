# LinearRegression is a machine learning library for linear regression
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np
# from __future__ import division, print_function, unicode_literals
# from datetime import datetime, date
# import yfinance as yf
# import streamlit as st
# from sklearn.metrics import r2_score
# from sklearn import datasets, linear_model

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('seaborn-darkgrid')

# yahoo finance is used to fetch data
# df = pd.read_csv('TapHuanLuyen.csv')
# Read data
Df = pd.read_csv('TapHuanLuyen.csv')


Df = Df[['Close']]


Df = Df.dropna()

# biểu đồ giá đóng của vàng SCJ
Df.Close.plot(figsize=(10, 7), color='r')
plt.ylabel("Giá vàng SCJ")
plt.title("Dãy giá vàng SCJ")
plt.show()

# Define explanatory variables
Df['Open'] = Df['Close'].rolling(window=3).mean()
Df['Close'] = Df['Close'].rolling(window=9).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['Open', 'Close']]

# Define dependent variable
y = Df['next_day_price']

t = .8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]

# Create a linear regression model
linear = LinearRegression().fit(X_train, y_train)
print("Linear Regression model")
print("Gold SCJ Price (y) = %.2f * 3 Days Moving Average (x1) \
+ %.2f * 9 Days Moving Average (x2) \
+ %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))

# Dự đoán giá vàng SCJ
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 7))
y_test.plot()
plt.legend(['Giá dự đoán', 'Giá thực tế'])
plt.ylabel("Giá vàng SCJ")
plt.show()


# R square
r2_score = linear.score(X[t:], y[t:])*100
float("{0:.2f}".format(r2_score))

gold = pd.DataFrame()


gold['price'] = Df[t:]['Close']
gold['predicted_price_next_day'] = predicted_price
gold['actual_price_next_day'] = y_test
gold['gold_returns'] = gold['price'].pct_change().shift(-1)

gold['signal'] = np.where(gold.predicted_price_next_day.shift(
    1) < gold.predicted_price_next_day, 1, 0)

gold['strategy_returns'] = gold.signal * gold['gold_returns']
((gold['strategy_returns']+1).cumprod()).plot(figsize=(10, 7), color='g')
plt.ylabel('Lợi nhuận tích lũy')
plt.show()


# # Build Xbar
# one = np.ones((X2.shape[0], 1))
# Xbar = np.concatenate((one, X1, X2, X3, X4, X5), axis=1)


# # fit the model by Linear Regression
# # fit_intercept = False for calculating the bias
# regr = linear_model.LinearRegression(fit_intercept=False)
# regr.fit(Xbar, y)

# w = regr.coef_
# w = w.T

# # print( 'Solution found by scikit-learn  : ', w )
# w_0 = w[0][0]
# w_1 = w[1][0]
# w_2 = w[2][0]
# w_3 = w[3][0]
# w_4 = w[4][0]
# w_5 = w[5][0]


# df1 = pd.read_csv('Test.csv')


# Test_1 = np.array(df1[['Volume']])


# Test_2 = np.array(df1[['Open']])


# Test_3 = np.array(df1[['High']])


# Test_4 = np.array(df1[['Low']])


# Test_5 = np.array(df1[['Adj close']])


# z = w_0 + w_1*Test_1 + w_2*Test_2 + w_3*Test_3 + w_4*Test_4 + w_5*Test_5

# print('Prediction: ', z)

# df1['Prediction'] = z

# price = np.array(
#     df1.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Market Cap'], axis=1))

# file = st.file_uploader("Pick a file")

# if file:
#     df1 = pd.read_csv(file)

#     Test_1 = np.array(df1[['Volume']])

#     Test_2 = np.array(df1[['Open']])

#     Test_3 = np.array(df1[['High']])

#     Test_4 = np.array(df1[['Low']])

#     Test_5 = np.array(df1[['Adj close']])

#     Test_6 = np.array(df1[['Close']])

#     if st.button('Dự đoán'):
#         z = w_0 + w_1*Test_1 + w_2*Test_2 + w_3*Test_3 + w_4*Test_4 + w_5*Test_5

#         st.write("Độ chính xác: ", r2_score(Test_6, z))

#         df1['Prediction'] = z
#         st.dataframe(df1)   # output ra bảng

data = pd.read_csv('TapHuanLuyen.csv')
data['S_3'] = data['Close'].rolling(window=3).mean()
data['S_9'] = data['Close'].rolling(window=9).mean()
data = data.dropna()
data['predicted_gold_price'] = linear.predict(data[['S_3', 'S_9']])
data['signal'] = np.where(data.predicted_gold_price.shift(
    1) < data.predicted_gold_price, "Buy", "No Position")
data.tail(7)
