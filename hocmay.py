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

# yahoo finance is used to fetch data

# Đọc dữ liệu
Df = pd.read_csv("test.csv")
# Df= pd.read_csv("TapHuanLuyen1.csv")
# Df = Df.date('2020-02-02','2020-11-12')
# Giữ các
Df = Df[['Price']]

# Drop rows with missing values
Df = Df.dropna()

# Vẽ biểu đồ giá đóng của GLD
# Df.Close.plot(figsize=(10, 7), color='r')
plt.figure(figsize=(16, 8))
plt.xlabel('Date')
plt.ylabel("Giá vàng ETF")
st.title("Biểu đồ giá vàng ETF")
plt.plot(Df['Price'])
plt.show()
st.pyplot()


Df = pd.read_csv("TapHuanLuyen1.csv")
# Xác định các biến giải thích
Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['next_day_price'] = Df['Close'].shift(-1)


Df = Df.dropna()
X = Df[['Close', 'S_3', 'S_9']]

# Xác định biến phụ thuộc
y = Df['next_day_price']


# Tách tập dữ liệu thành tập huấn luyện và thử nghiệm
t = .8
t = int(t*len(Df))

# Tập dữ liệu để đào tạo Train
X_train = X[:t]
y_train = y[:t]

st.write(X_train)
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
plt.ylabel("Giá vàng ETF")
plt.show()

st.title("Mô hình dự đoán giá dự đoán với giá thực tế")
st.pyplot()

#  R square (Độ chính xác về dự đoán)
r2_score = linear.score(X[t:], y[t:])
print(float("{0:.2f}".format(r2_score)))
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
plt.ylabel('Lợi nhuận tích lũy')
plt.show()

st.title("Dự đoán lợi nhuận")
st.pyplot()
# Tỷ lệ sharpe
# print('Sharpe Ratio %.2f' % (gold['strategy_returns'].mean(
# )/gold['strategy_returns'].std()*(252**0.5)))

# Sử dụng mô hình để dự đoán giá chuyển động hằng ngày

st.title("Theo dõi chuyển động giá hàng ngày để đưa ra quyết định nên giao dịch hay không")


st.write(Df.tail(10))
st.write("Giá vàng trong 10 ngày gần nhất")

# Đoạn code sau để dự đoán giá vàng và đưa ra tín hiệu giao dịch xem mọi người có nên mua GLD hay không
# data = yf.download('GLD', '2020-01-01', '2021-4-17', auto_adjust=True)
data = pd.read_csv("TapHuanLuyen1.csv")
data['S_3'] = data['Close'].rolling(window=3).mean()
data['S_9'] = data['Close'].rolling(window=9).mean()
data = data.dropna()
data['predicted_gold_price'] = linear.predict(data[['Close', 'S_3', 'S_9']])
data['signal'] = np.where(data.predicted_gold_price.shift(
    1) < data.predicted_gold_price, "Buy", "No Position")
print(data.tail(7))
st.write((data['predicted_gold_price']).tail(10))

st.write("Bảng dự đoán quyết định giao dịch : ")
st.write((data['signal']).tail(10))

st.write("Dự đoán tập test")

X_1 = pd.read_csv("TrainingSet.csv")
st.write(X_1)
st.write("Kết quả :")
Y_1 = linear.predict(X_1)
st.write(Y_1)


st.sidebar.title('Dự đoán giá vàng ngày mai')
S_in = st.sidebar.text_input('Nhập giá vàng hôm nay')
S_3in = st.sidebar.text_input('Nhập giá vàng trung bình 3 ngày')
S_9in = st.sidebar.text_input('Nhập giá vàng trung bình 9 ngày ')
inp = np.array([[S_in, S_3in, S_9in]])


if st.sidebar.button('Dự đoán'):
    Y_2 = linear.predict(inp)
    st.write('Giá vàng ngày mai dự đoán là :', Y_2, '$')
