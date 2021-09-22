from __future__ import division, print_function, unicode_literals
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Mô hình dự đoán giá nhà đất tại hồ gươm ')

# x1 là diện tích của lô đất(m2)
# x2 là chiều dài mặt tiền (m)
# x3 là số tầng nhà
# x4 là khoảng cách tới hồ gươm (m)
X = np.array([[40, 8, 2, 1800],
              [36, 3.5, 6, 450],
              [35, 4.5, 6, 450],
              [39, 9, 2, 1800],
              [40, 9, 1, 1800],
              [36, 4.5, 5, 450],
              [36, 4.5, 6, 450],
              [40, 9, 2, 1800],
              [36, 4.5, 7, 450],
              [40, 9, 3, 1800],
              [44, 4, 5, 350],
              [41, 9, 2, 1800],
              [37, 4.5, 6, 450],
              [36, 5.5, 6, 450],
              [40, 10, 2, 1800],
              [45, 3, 4, 350],
              [45, 4, 3, 350],
              [45, 4, 4, 350],
              [45, 4, 5, 350],
              [45, 5, 4, 350],
              [45, 3, 4, 350],
              [60, 2.3, 5, 450],
              [59, 3.3, 5, 450],
              [60, 3.3, 4, 450],
              [85, 4, 4, 950],
              [85, 4, 5, 950],
              [60, 3.3, 5, 450],
              [61, 6, 1, 800],
              [62, 5, 1, 800],
              [85, 4, 6, 950],
              [84, 6, 5, 950],
              [86, 2.5, 3, 900],
              [60, 3.3, 6, 450],
              [85, 5, 5, 950],
              [85, 3.5, 3, 900],
              [86, 3.5, 2, 900],
              [31.2, 3, 4, 450],
              [61, 3.3, 5, 450],
              [62, 6, 1, 800],
              [85, 6, 5, 950],
              [86, 3.5, 3, 900],
              [62, 6, 2, 800],
              [86, 3.5, 4, 900],
              [87, 3.5, 3, 900],
              [30.2, 4, 4, 450],
              [62, 6, 3, 800],
              [86, 4.5, 3, 900],
              [86, 6, 5, 950],
              [60, 4.3, 5, 450],
              [62, 7, 1, 800],
              [63, 6, 1, 800],
              [31.2, 4, 4, 450],
              [31.2, 4, 3, 450],
              [62, 4, 5, 550],
              [31.2, 4, 5, 450],
              [63, 5, 3, 550],
              [63, 4, 5, 550],
              [32.2, 4	, 4, 450],
              [31.2, 5, 4, 450],
              [63, 5, 5, 550],
              [64, 4, 5, 550],
              [63, 5, 6	, 550],
              [63, 6, 4, 550],
              [80, 5.8, 7, 1100],
              [80, 4.8, 8, 1100],
              [80, 5.8, 8, 1100],
              [79, 5.8, 8, 1100],
              [80, 5.8, 9, 1100],
              [81, 5.8, 8, 1100],
              [80, 6.8, 8, 1100],
              [80, 3.5, 6, 300],
              [80, 4.5, 5, 300],
              [80, 4.5, 6, 300],
              [79, 4.5, 6, 300],
              [81, 4.5, 6, 300],
              [88, 3.5, 4, 850],
              [88, 4.5, 3, 850],
              [88, 4.5, 4, 850],
              [87, 4.5, 4, 850],
              [88, 4.5, 5, 850],
              [89, 4.5, 4, 850],
              [88, 5.5, 4, 850],
              [80, 5.5, 7, 300],
              [63, 6, 4, 250],
              [62, 7, 4, 250],
              [63, 7, 3, 250],
              [63, 7, 4, 250],
              [63, 7, 5, 250],
              [64, 7, 4, 250],
              [63, 8, 4, 250],
              [140, 4.5, 5, 500],
              [139, 5.5, 5, 500],
              [140, 5.5, 4, 500],
              [140, 5.5, 5, 500],
              [140, 5.5, 6, 500],
              [141, 5.5, 5, 500],
              [140, 6.5, 5, 500]])
Y = np.array([[
    19, 19.3, 19.45, 19.48, 19.5, 19.7, 20, 20, 20.3, 20.5,
    20.5, 20.52, 20.55, 20.7, 21, 21, 21.3, 21.5, 21.7, 22,
    22.5, 29, 30, 30.5, 30.5, 30.8, 31, 31, 31, 31, 31.3, 31.35,
    31.5, 31.5, 31.63, 31.7, 32, 32, 32, 32, 32, 32.3, 32.3, 32.37,
    32.4, 32.5, 32.65, 32.7, 33, 33, 33, 33.5, 33.5, 33.6, 34, 34, 34.3,
    34.6, 35, 35, 35, 35.5, 35.7, 42.5, 42.9, 43, 43.463, 43.5, 43.537,
    44.1, 50, 52.3, 53, 53.38, 53.62, 54, 54.5, 55, 55.46, 55.5, 55.54, 56,
    56.7, 60, 62.3, 62.5, 63, 63.5, 63.7, 66, 96.5, 97.3, 97.5, 98, 98.5, 98.7, 99.5
]]).T


def duel_plot(X1, X2, Y):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(Y, X[:, 0])
    ax1.set_title('xét diện tích với giá tiền')
    ax1.set_xlabel('giá tiền')
    ax1.set_ylabel('Diện tích m2')

    ax2.plot(Y, X[:, 1])
    ax2.set_title('xét số mét mặt tiền với giá tiền')
    ax2.set_xlabel('giá tiền')
    ax2.set_ylabel('số mét mặt tiền')
    return fig


def duel_plot2(X4, X5, Y):

    fig = plt.figure(figsize=(15, 5))
    ax3 = fig.add_subplot(1, 2, 1)
    ax4 = fig.add_subplot(1, 2, 2)
    ax3.plot(Y, X[:, 2])
    ax3.set_title('xét số tầng nhà với giá tiền')
    ax3.set_xlabel('giá tiền')
    ax3.set_ylabel('số tầng nhà')

    ax4.plot(Y, X[:, 3])
    ax4.set_title('xét khoảng cách với giá tiền')
    ax4.set_xlabel('giá tiền')
    ax4.set_ylabel('khoảng cách tới hồ gươm')
    return fig


st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(duel_plot(X[:, 0], X[:, 1], Y))
st.pyplot(duel_plot2(X[:, 2], X[:, 3], Y))


st.sidebar.title('Dự đoán giá các mẫu nhà')
dt_name = st.sidebar.text_input('Nhập diện tích đất(m2) ')
cd_name = st.sidebar.text_input('Nhập chiều dài mặt tiền(m) ')
tn_name = st.sidebar.text_input('Nhập số tầng nhà(tầng) ')
kc_name = st.sidebar.text_input('Nhập khoảng cách nhà tới hồ gươm(m) ')
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

x_train, x_test, y_train, y_test = train_test_split(Xbar, Y, test_size=0.2)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y)
w = np.dot(np.linalg.pinv(A), b)


w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]

st.write("Độ chính xác (R2 square) : ", r2_score(y_test, np.dot(x_test, w)))

vd = np.array([dt_name, cd_name, tn_name, kc_name, 1])
if st.sidebar.button('Dự đoán'):
    y1 = w_1*float(dt_name)+w_2*float(cd_name)+w_3 * \
        float(tn_name)+w_4*float(kc_name) + w_0
    st.sidebar.write('Giá của ngôi nhà là : ', y1, 'tỷ đồng')
