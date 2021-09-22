from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tkinter import *
X1 = [[2+3.37319011, 1+3.71875981],
    [1+3.51261889, 1+3.40558943],
    [2+3.4696794, 2+3.02144973],
    [1+3.78736889, 1+3.29380961],
    [1+3.81231157, 1+3.56119497],
    [2+3.03717355, 1+3.93397133],
    [1+3.53790057, 1+3.87434722],
    [2+3.29312867, 2+3.76537389],
    [1+3.38805594, 1+3.86419379],
    [1+3.57279694, 8.90707347]]
y1 = [1, 1, 1, 1, 1, 1, 1, 1,1,1]

X2 = [[3+3.42746579, 3.71254431],
    [4+3.24760864, 2+3.39846497],
    [3+3.33595491, 1+3.61731637],
    [3+3.69420104, 1+3.94273986],
    [4+3.53897645, 2+3.54957308],
    [3+3.3071994, 3.19362396],
    [4+3.13924705, 2+3.09561534],
    [4+3.47383468, 2+3.41269466],
    [4+3.00512009, 1+3.89290099],
    [4+3.28205624, 1+3.79675607]]
y2 = [-1, -1, -1, -1, -1, -1, -1, -1,-1,-1]
X = np.array(X1 + X2)
y = y1 + y2

clf = SVC(kernel='linear', C=1E10)
clf.fit(X, y)

def doan():
    a = entry1.get()
    b = entry2.get()
    X_pr =[[a,b]]
    res = clf.predict(X_pr)
    label_show.set(res)

root =Tk()
root.option_add("*Font","TimeNewRoman 20")

label_show=StringVar()

Label (root, text="Hãy nhập X và Y").grid(row=0,columnspan=2)
Label (root, text="Nhập X ").grid(row=1,column=0,padx=10,pady=10)
entry1= Entry(root) 
entry1.grid(row=1,column=1)

Label (root, text="Nhập Y ").grid(row=2,column=0,padx=10,pady=10)
entry2= Entry(root)
entry2.grid(row=2,column=1)

Button (root, text="Kết quả dự đoán", command=doan,bg= 'white').grid(row=3,column=0,padx=10,pady=10)
Label(root,textvariable=label_show).grid(row=3,column=1)

root.mainloop()