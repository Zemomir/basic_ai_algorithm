import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv(r"C:\Users\emir\Desktop\zart zurt projelerim\Salary Data.csv")
x = np.array(dataset["YearsExperience"])
y = np.array(dataset["Salary"])

w=1
b=1
lr= 0.017
epoch = 1027
#31296750
""" Batch Gradient Descent"""
for i in range(epoch):
    y_hat = w*x + b
    J = ((y-y_hat)**2).sum()/len(x)
    dw = -2*(x*(y-y_hat)).sum()/len(x)
    db = -2*(y-y_hat).sum()/len(x)
    w = w -dw*lr
    b = b -db*lr
plt.scatter(x,y) 
plt.plot(x,y_hat)
plt.show()
print(f"Batch Gradient Descentin 1.75 teki sonucu {1.75*w+b}")
""" Stochastic Gradient Descent"""
w=1
b=1
lr=0.001
epoch=100000
for i in range(epoch):
    for n in range(len(x)):
        y_hat = w*x[n] +b
        J = (y[n]-y_hat)**2
        dw = -2*(y[n]-y_hat)*x[n]
        db = -2*(y[n]-y_hat)
        w = w -dw*lr
        b = b -db*lr
y_hat = w *x +b
plt.scatter(x,y) 
plt.plot(x,y_hat)
plt.show()
print(f"stochastic Gradient Descentin 1.75 teki sonucu {1.75*w+b}")
