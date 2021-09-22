# LinearRegression is a machine learning library for linear regression
# Nhập thư viện và đọc dữ liệu giá vàng
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('seaborn-darkgrid')

# Đọc dữ liệu
Df = yf.download('GLD', '2020-01-01', '2020-4-17', auto_adjust=True)

# Giữ các
Df = Df[['Close']]

# Drop rows with missing values
Df = Df.dropna()

# Vẽ biểu đồ giá đóng của GLD
Df.Close.plot(figsize=(10, 7), color='r')
plt.ylabel("Gold ETF Prices")
plt.title("Gold ETF Price Series")
plt.show()
st.pyplot()

# Xác định các biến giải thích
Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S_3', 'S_9']]

# Xác định biến phụ thuộc
y = Df['next_day_price']


# Tách tập dữ liệu thành tập huấn luyện và thử nghiệm
t = .8
t = int(t*len(Df))

# Tập dữ liệu để đào tạo Train
X_train = X[:t]
y_train = y[:t]

# Bộ dữ liệu thử nghiệm
X_test = X[t:]
y_test = y[t:]


# Tạo mô hình hồi quy tuyến tính
linear = LinearRegression().fit(X_train, y_train)
print("Mô hình hồi quy tuyến tính")
print("Giá vàng ETF (y) = %.2f * Trung bình cộng 3 ngày (x1) \
+ %.2f * Trung bình cộng trong 9 ngày (x2) \
+ %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))


# Dự đoán giá vàng ETF
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 7))
y_test.plot()
plt.legend(['Giá dự đoán', 'Giá thực tế'])
plt.ylabel("Gía vàng ETF")
plt.show()
st.pyplot()

y_pred = linear.predict(X_test)

#  R square (Độ chính xác về dự đoán)
r2_score = linear.score(X_test, y_pred)
print(float("{0:.2f}".format(r2_score)))
st.write(X[t:])
st.write(y_pred)
st.write("Score : ", r2_score)

gold = pd.DataFrame()

gold['price'] = Df[t:]['Close']
gold['predicted_price_next_day'] = predicted_price
gold['actual_price_next_day'] = y_test
gold['gold_returns'] = gold['price'].pct_change().shift(-1)

gold['signal'] = np.where(gold.predicted_price_next_day.shift(
    1) < gold.predicted_price_next_day, 1, 0)

gold['strategy_returns'] = gold.signal * gold['gold_returns']
((gold['strategy_returns']+1).cumprod()).plot(figsize=(10, 7), color='g')
plt.ylabel('Cumulative Returns')
plt.show()
st.pyplot()

'Sharpe Ratio %.2f' % (gold['strategy_returns'].mean(
)/gold['strategy_returns'].std()*(252**0.5))


data = yf.download('GLD', '2008-06-01', '2020-6-25', auto_adjust=True)
data['S_3'] = data['Close'].rolling(window=3).mean()
data['S_9'] = data['Close'].rolling(window=9).mean()
data = data.dropna()
data['predicted_gold_price'] = linear.predict(data[['S_3', 'S_9']])
data['signal'] = np.where(data.predicted_gold_price.shift(
    1) < data.predicted_gold_price, "Buy", "No Position")
print(data.tail(7))
