from __future__ import division, print_function, unicode_literals
import streamlit as st
from sklearn.metrics import mean_squared_error , r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Dự đoán giá nhà đất tại hồ gươm ')

#x1 là diện tích của lô đất(m2)
#x2 là chiều dài mặt tiền (m)
#x3 là số tầng nhà
#x4 là khoảng cách tới hồ gươm (m)
X = np.array([[31.2],
             [110],
             [55], 
             [268],
             [650]])
y = np.array([[33.5,35.95,60,268,346]]).T
x1 = [4,23,4.5,5,25]
x2 = [4,4,5,3,3]
x3 = [700,400,400,1000,280]
 #y là giá tiền đơn vị là tỷ đồng
def duel_plot(X,x1,y) :
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(X,y)
    ax1.set_title('xet x với y')
    ax1.set_xlabel('dien tich')
    ax1.set_ylabel('gia tien')

    ax2.plot(x1, y)
    ax2.set_title('xet y_true với y')
    ax2.set_xlabel('khoảng cách')
    ax2.set_ylabel('giá tiền')
    return fig
def duel_plot2(x2,x3,y):
    fig = plt.figure(figsize=(15,5))
    ax3 = fig.add_subplot(1, 2, 1)
    ax4 = fig.add_subplot(1, 2, 2)
    ax3.plot(x2, y)
    ax3.set_title('xet x2 với y')
    ax3.set_xlabel('x2')
    ax3.set_ylabel('giá tiền')

    ax4.plot(x3, y)
    ax4.set_title('xet x3 với y')
    ax4.set_xlabel('x3')
    ax4.set_ylabel('giá tiền')
    return fig
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(duel_plot(X,x1,y))
st.pyplot(duel_plot2(x2,x3,y))




st.title('Dự đoán giá các mẫu nhà')
dt_name = st.text_input('Nhập diện tích đất(m2) ')
cd_name = st.text_input('Nhập chiều dài mặt tiền(m) ')
tn_name = st.text_input('Nhập số tầng nhà(tầng) ' )
kc_name = st.text_input('Nhập khoảng cách nhà tới hồ gươm(m) ' )
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)


w = np.dot(np.linalg.pinv(A), b)


w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]


vd = np.array([dt_name,cd_name,tn_name,kc_name,1])

y1 = w_1*int(dt_name)+w_2*int(cd_name)+w_3*int(tn_name)+w_4*int(kc_name)+ w_0
st.write('Giá của ngôi nhà là : ', y1, 'tỷ đồng')


