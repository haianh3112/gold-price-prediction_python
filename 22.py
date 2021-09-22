import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import datetime
import warnings
warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')

df = pd.read_csv('Covid-dataset.csv')
df.drop(df[df.location == 'World'].index, inplace=True)
df.drop(df[df.location == 'Africa'].index, inplace=True)
df.drop(df[df.location == 'Asia'].index, inplace=True)
df.drop(df[df.location == 'Europe'].index, inplace=True)
df.drop(df[df.location == 'European'].index, inplace=True)
df.drop(df[df.location == 'International'].index, inplace=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
covid = df[df['date'] >= '2020-08-01']

covid["total_vaccinations"].fillna(0, inplace=True)
# replace smokers
covid['smokers'] = (covid['female_smokers'] + covid['male_smokers']) / 100 * 50
covid.drop('female_smokers', inplace=True, axis=1)
covid.drop('male_smokers', inplace=True, axis=1)
covid['temp'] = covid['new_cases']
covid['new_cases'] = covid['smokers']
covid['smokers'] = covid['temp']
covid.drop(columns=['temp'], inplace=True)
covid = covid.rename(columns={"new_cases": "smokers", "smokers": "new_cases"})
covid['smokers'].fillna(covid['smokers'].mean(), inplace=True)
covid['new_cases'].fillna(method='ffill', inplace=True)

covid.dropna(subset=['population_density', 'median_age',
             'stringency_index', 'positive_rate'], inplace=True)

unique_country = pd.unique(covid['location'])
for i in unique_country:
    if sum(covid.loc[covid['location'] == i, 'new_cases']) < 300000:
        covid.drop(covid[covid.location == i].index, inplace=True)

dates = []
d = datetime.datetime(2020, 8, 1)
while d < datetime.datetime(2021, 3, 25):
    d += datetime.timedelta(days=1)
    dates.append(d)

i = 0
while i < len(dates):
    sum_cases = sum(covid.loc[covid['date'] == dates[i], 'new_cases'])
    if sum_cases >= 650000 or dates[i] >= datetime.datetime(2020, 12, 1) and dates[i] <= datetime.datetime(2021, 1, 1) and sum_cases <= 430000:
        covid.drop(covid.index[covid['date'] == dates[i]], inplace=True)
        dates.remove(dates[i])
    else:
        i += 1

X = covid.iloc[:, 3:-1]
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
y = covid.iloc[:, -1]

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X = min_max_scaler.fit_transform(X)
poly_reg = PolynomialFeatures(degree=3)
X = poly_reg.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Polynomial Regression (degree=3)
A = np.dot(x_train.T, x_train)
b = np.dot(x_train.T, y_train)
W = np.dot(np.linalg.pinv(A), b)
y_test_predict = np.dot(x_test, W)
y_predict = np.dot(X, W)

daily_cases = []
for d in dates:
    sum_cases = sum(covid.loc[covid['date'] == d, 'new_cases'])
    daily_cases.append(sum_cases)

predict_covid = pd.DataFrame(covid.loc[:, 'date'])
predict_covid['new_cases_predict'] = y_predict
daily_cases_predict = []
for d in dates:
    sum_cases_predict = sum(
        predict_covid.loc[predict_covid['date'] == d, 'new_cases_predict'])
    daily_cases_predict.append(sum_cases_predict)


st.title("Covid-19 Cases Prediction")
st.write("Covid-19 Dataset")
st.write(covid)
st.write("Shape of dataset (9383, 11)")
st.write("Algorithm : Polynomial Regression, degree = 3")
st.write("Hệ số W : ", W)
st.write("Đánh giá mô hình r2_score : ", r2_score(y_test, y_test_predict))
fig = plt.figure(figsize=(10, 6))
plt.xlabel('Date', fontsize=20)
plt.xticks(rotation=45)
plt.ylabel('New cases', fontsize=20)
plt.scatter(dates, daily_cases)
plt.plot(dates, daily_cases_predict, color='coral', linewidth=3)
plt.show()
st.pyplot(fig)

st.sidebar.title("Input values \n")
population_dens = st.sidebar.text_input('Mật độ dân số (Người / km2)')
median_age = st.sidebar.text_input('Độ tuổi trung bình')
total_vacci = st.sidebar.text_input('Tổng số liều vaccine cung cấp')
diabets = st.sidebar.text_input('Tỉ lệ mắc bệnh tiểu đường')
stringency_index = st.sidebar.text_input(
    'Chỉ số phản ứng của chính phủ')
positive_rate = st.sidebar.text_input('Tỉ lệ dương tính trong ngày')
smokers = st.sidebar.text_input('Tỉ lệ hút thuốc')
submit = st.sidebar.button('Predict')

if submit:
    inputs = np.array([[population_dens, median_age, total_vacci, diabets, stringency_index,
                        positive_rate, smokers]])
    inputs = np.concatenate((np.ones((1, 1)), inputs), axis=1)
    inputs = min_max_scaler.fit_transform(inputs)
    inputs = poly_reg.fit_transform(inputs)
    st.write("Số ca mắc : ", np.dot(inputs, W))
