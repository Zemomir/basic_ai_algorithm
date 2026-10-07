import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\emir\Desktop\projeler veri setleri\zart zurt projelerim\Salary Data.csv")
x = data["YearsExperience"]
y= data["Salary"]
x= np.array(x)
y = np.array(y)
def mae_lr(x,y):
    w = np.random.uniform(-1,1)
    b = np.random.uniform(-1,1)
    lr = 0.1
    epoch = 850
    for i in range(epoch):
        y_hat = w*x + b
        cost = np.abs(y_hat - y)
        print(f"MAE cost = {cost.mean()}")
        grad = np.sign(cost)
        #subgradient
        grad[cost==0] = np.random.uniform(-1,1,size= (cost==0).sum())
        #gradient
        dw = - np.mean(np.sum(grad*x))
        db = - np.mean(np.sum(grad))
        #update
        w -= dw*lr
        b -= db*lr

    return w,b
def mse_lr(x,y):
    w = np.random.uniform(-1,1)
    b = np.random.uniform(-1,1)
    lr = 0.0001
    epoch = 850
    for i in range(epoch):
        y_hat = w*x + b
        cost = (y-y_hat)**2
        print(f"MSE cost = {cost.mean()}")
        #gradient
        dw = - 2*np.mean(np.sum((y-y_hat)*x))
        db = - 2*np.mean(np.sum(y-y_hat))
        #update
        w -= dw*lr
        b -= db*lr

    return w,b
plt.scatter(x,y,c="r")
w,b = mae_lr(x,y)
plt.plot(x,w*x+b,c="b")

w,b = mse_lr(x,y)
plt.plot(x,w*x+b,c="g")

plt.show()