from __future__ import division, print_function, unicode_literals
import pandas as pd
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score


df = pd.read_csv("TapHuanLuyen.csv",index_col=None,sep=',')



       
X = df.iloc[:, 1:4]
X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
y = df.iloc[:,-3]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

A = np.dot(x_train.T, x_train)
b = np.dot(x_train.T, y_train)
W = np.dot(np.linalg.pinv(A), b)
y_test_predict = np.dot(x_test, W)
y_predict = np.dot(X, W)

st.title("Dự đoán giá vàng")
st.text_input("Open")
st.text_input("High")
st.text_input("Low")
st.write("Hệ số W : ", W)
st.write("Đánh giá mô hình r2_score : ", r2_score(y_test, y_test_predict))

fig = plt.figure(figsize=(10,6))
plt.xlabel('Giá vàng thực tế')
plt.ylabel('Giá vàng dự đoán')
plt.scatter(y_test,y_test_predict)
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()
st.pyplot(fig)

st.sidebar.title("Input values \n")
Open = st.sidebar.text_input('Open1') 
High = st.sidebar.text_input('High1')
Low = st.sidebar.text_input('Low1')

submit = st.sidebar.button('Predict')

if submit: 
    inputs = np.array([[Open, High, Low,]])
    inputs = np.concatenate((np.ones((1,1)), inputs), axis=1)
    st.write("Giá Close : ", np.dot(inputs, W))
