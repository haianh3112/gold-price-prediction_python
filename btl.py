import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


data_df = pd.read_csv('test.csv')
data_df.head()


x = data_df.drop(['Price'], axis=1).values
y = data_df['Price'].values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

ml = LinearRegression()
ml.fit(x_train, y_train)

y_pred = ml.predict(x_test)
print(y_pred)

print(ml.predict([[1, 3.5, 800, 4]]))

print(r2_score(y_test, y_pred))

plt.figure(figsize=(15, 10))
plt.scatter(y_test, y_pred)
plt.axis([1000000, 40000000, 1000000, 40000000])
plt.xlabel('Thực tế')
plt.ylabel('Dự đoán')
plt.title('Thực tế và dự đoán')
plt.show()
