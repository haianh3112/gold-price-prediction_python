# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
from datetime import datetime, date
import yfinance as yf
import streamlit as st
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('C:\Downloads\TapHuanLuyen.csv')


X1 = np.array(df[['Volume']])


X2 = np.array(df[['Open']])


X3 = np.array(df[['High']])


X4 = np.array(df[['Low']])

X5 = np.array(df[['Market Cap']])

# Lấy cột Close (Giá của một ngày)
y = np.array(
    df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Market Cap'], axis=1))

# Build Xbar
one = np.ones((X2.shape[0], 1))
Xbar = np.concatenate((one, X1, X2, X3, X4, X5), axis=1)


# fit the model by Linear Regression
# fit_intercept = False for calculating the bias
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)

w = regr.coef_
w = w.T

# print( 'Solution found by scikit-learn  : ', w )

w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]

# ## Load test data
# df1 = pd.read_csv('Test.csv')

# # Lấy cột Volume
# Test_1 = np.array(df1[['Volume']])

# # Lấy cột Open
# Test_2 = np.array(df1[['Open']])

# # Lấy cột High
# Test_3 = np.array(df1[['High']])

# # Lấy cột Low
# Test_4 = np.array(df1[['Low']])

# # Lấy cột Market Cap
# Test_5 = np.array(df1[['Market Cap']])


# z = w_0 + w_1*Test_1 + w_2*Test_2 + w_3*Test_3 + w_4*Test_4 + w_5*Test_5

# # print('Prediction: ', z)

# df1['Prediction'] = z

# price = np.array(df1.drop(['Date','Open','High','Low','Volume','Market Cap'], axis = 1))

# print(price)


# GUI

st.write("""
# Dự đoán giá Bitcoin
""")

st.write("""
## Giá mở
""")
st.line_chart(df.Open)

st.write("""
## Giá đóng
""")
st.line_chart(df.Close)

st.write("""
## Số lượng giao dịch trong ngày
""")
st.area_chart(df.Volume)

file = st.file_uploader("Pick a file")

if file:
    df1 = pd.read_csv(file)
    # Lấy cột Volume
    Test_1 = np.array(df1[['Volume']])

    # Lấy cột Open
    Test_2 = np.array(df1[['Open']])

    # Lấy cột High
    Test_3 = np.array(df1[['High']])

    # Lấy cột Low
    Test_4 = np.array(df1[['Low']])

    # Lấy cột Market Cap
    Test_5 = np.array(df1[['Market Cap']])

    Test_6 = np.array(df1[['Close']])

    if st.button('Dự đoán'):
        z = w_0 + w_1*Test_1 + w_2*Test_2 + w_3*Test_3 + w_4*Test_4 + w_5*Test_5

        st.write("Độ chính xác: ", r2_score(Test_6, z))   # Accuracy của LR

        df1['Prediction'] = z       # Thêm một cột Prediction
        st.dataframe(df1)   # output ra bảng
        # Xử lý dữ liệu in ra biểu đồ
        price = pd.DataFrame(
            df1.drop(['Open', 'High', 'Low', 'Volume', 'Market Cap'], axis=1))
        price = price.rename(columns={'Date': 'index'}).set_index('index')
        st.write("""
        ## So sánh giá Close và giá Prediction
        """)
        st.line_chart(price)
