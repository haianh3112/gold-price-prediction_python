import streamlit as st
import numpy as np

st.title('Dự đoán giá nhà đất tại hồ gươm ')
st.title('Dự đoán giá các mẫu nhà')
dt_name = st.text_input('Nhập diện tích đất(m2) ')
cd_name = st.text_input('Nhập chiều dài mặt tiền(m) ')
tn_name = st.text_input('Nhập số tầng nhà(tầng) ')
kc_name = st.text_input('Nhập khoảng cách nhà tới hồ gươm(m) ')
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
